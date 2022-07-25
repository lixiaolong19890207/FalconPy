
__constant__ float const_transform_matrix[9];

cudaArray* d_volume_array = 0;
cudaExtent volume_size;

unsigned char* p_vr = 0;
float3 normal, spacing, max_per;
int volume_of_interest[6];
int width_vr = 0;
int height_vr = 0;


__device__ float3 mul(const float &m, const float3 &v)
{
	float3 r;
	r.x = dot(v, M.m[0]);
	r.y = dot(v, M.m[1]);
	r.z = dot(v, M.m[2]);
	return r;
}

__device__ uint rgba_float_to_int(float4 rgba)
{
	rgba.x = __saturatef(rgba.x);
	rgba.y = __saturatef(rgba.y);
	rgba.z = __saturatef(rgba.z);
	rgba.w = 0.0f;
	return (uint(rgba.z*255)<<16) | (uint(rgba.y*255)<<8) | uint(rgba.x*255);
}

__device__ float4 tracing(
	float4 sum,
	float alphaAccObject,
	cudaTextureObject_t volumeText,
	float3 pos,
	float4 col,
	float3 dirLight,
	float3 f3Nor,
	bool invertZ
)
{
	float3 N;
	N.x = tex3D<float>(volumeText, pos.x+f3Nor.x, pos.y, pos.z) - tex3D<float>(volumeText, pos.x-f3Nor.x, pos.y, pos.z);
	N.y = tex3D<float>(volumeText, pos.x, pos.y+f3Nor.y, pos.z) - tex3D<float>(volumeText, pos.x, pos.y-f3Nor.y, pos.z);
	N.z = tex3D<float>(volumeText, pos.x, pos.y, pos.z+f3Nor.z) - tex3D<float>(volumeText, pos.x, pos.y, pos.z-f3Nor.z);
	if (invertZ){
		N.z = -N.z;
	}
	N = normalize(N);

	float diffuse = dot(N, dirLight);
	float4 clrLight = col * 0.35f;

	float4 f4Temp = make_float4(0.0f);
	if ( diffuse > 0.0f )
	{
		f4Temp = col * (diffuse*0.8f + 0.16f*(pow(diffuse, 8.0f)));
	}
	clrLight += f4Temp;

	diffuse = (1.0f - alphaAccObject) * col.w;
	return (sum + diffuse * clrLight);
}

__device__ bool getNextStep(
	float& fAlphaTemp,
	float& fStepTemp,
	float& accuLength,
	float fAlphaPre,
	float fStepL1,
	float fStepL4,
	float fStepL8
)
{
	if (fStepTemp == fStepL4)
		fAlphaTemp = 1 - pow(1-fAlphaTemp, 0.25f);
	else if(fStepTemp == fStepL8)
		fAlphaTemp = 1 - pow(1-fAlphaTemp, 0.125f);

	if (accuLength > 0.0f)
	{
		if (MAX(fAlphaTemp, fAlphaPre) > 0.001f)
		{
			if (fStepTemp == fStepL1)
			{
				accuLength -= (fStepL1 - fStepL4);
				fStepTemp = fStepL4;
				return false;
			}
			else if(fStepTemp == fStepL4)
			{
				accuLength -= (fStepL4 - fStepL8);
				fStepTemp = fStepL8;
				return false;
			}
		}
		else
		{
			if (fStepTemp == fStepL8)
				fStepTemp = fStepL4;
			else
				fStepTemp = fStepL1;
		}
	}
	return true;
}

/*
**   z
**   |__x
**  /-y
*/

__global__ void d_render(
	unsigned char* pPixelData,
	cudaTextureObject_t volumeText,
	int width,
	int height,
	float xTranslate,
	float yTranslate,
	float scale,
	float3 f3maxper,
	float3 f3Spacing,
	float3 f3Nor,
	VOI voi,
	cudaExtent volumeSize,
	bool invertZ,
	float4 f4ColorBG
)
{
	const int x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	const int y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;

	if ((x < width) && (y < height) && (x >= 0) && (y >= 0))
	{
		uint nIdx = __umul24(y, width) + x;

		float u = 1.0f*(x-width/2.0f-xTranslate)/width;
		float v = 1.0f*(y-height/2.0f-yTranslate)/height;

		float4 sum = make_float4(0.0f);

		float3 dirLight = make_float3(0.0f, 1.0f, 0.0f);
		dirLight = normalize(mul(const_transform_matrix, dirLight));

		float fStepL1 = 1.0f/volumeSize.depth;
		float fStepL4 = fStepL1/4.0f;
		float fStepL8 = fStepL1/8.0f;
		float fStepTemp = fStepL1;

		float temp = 0.0f;
		float3 pos;

		float alphaAccObject[MAXOBJECTCOUNT+1];
		for (int i=0; i<MAXOBJECTCOUNT+1; i++){
			alphaAccObject[i] = 0.0f;
		}
		float alphaAcc = 0.0f;

		float accuLength = 0.0f;
		int nxIdx = 0;
		int nyIdx = 0;
		int nzIdx = 0;
		float fy = 0;

		float4 col;
		float fAlphaTemp = 0.0f;
		float fAlphaPre = 0.0f;

		unsigned char label = 0;
		float3 alphawwwl = make_float3(0.0f, 0.0f, 0.0f);

		while (accuLength < 1.732)
		{
			fy = (accuLength-0.866)*scale;

			pos = make_float3(u, fy, v);
			pos = mul(const_transform_matrix, pos);

			pos.x = pos.x * f3maxper.x + 0.5f;
			pos.y = pos.y * f3maxper.y + 0.5f;
			pos.z = pos.z * f3maxper.z + 0.5f;
			if (invertZ)
				pos.z = 1.0f - pos.z;

			nxIdx = pos.x * volumeSize.width;
			nyIdx = pos.y * volumeSize.height;
			nzIdx = pos.z * volumeSize.depth;
			if (nxIdx<voi.left || nxIdx>voi.right || nyIdx<voi.posterior || nyIdx>voi.anterior || nzIdx<voi.head || nzIdx>voi.foot)
			{
				accuLength += fStepTemp;
				continue;
			}
			label = 0;

			alphawwwl = constAlphaAndWWWL[label];

			temp = 32768*tex3D<float>(volumeText, pos.x, pos.y, pos.z);
			temp = (temp - alphawwwl.z)/alphawwwl.y + 0.5;
			if (temp>1)
				temp = 1;

			col = tex1D<float4>(constTransferFuncTexts[label], temp);

			fAlphaTemp = col.w;

			if (!getNextStep(fAlphaTemp, fStepTemp, accuLength, fAlphaPre, fStepL1, fStepL4, fStepL8)){
				continue;
			}

			fAlphaPre = fAlphaTemp;
			accuLength += fStepTemp;

			col.w = fAlphaTemp;

			if (col.w > 0.0005f && alphaAccObject[label] < alphawwwl.x){
				sum = tracing(sum, alphaAcc, volumeText, pos, col, dirLight, f3Nor, invertZ);
				alphaAccObject[label] += (1.0f - alphaAcc) * col.w;
				alphaAcc += (1.0f - alphaAcc) * col.w;
			}

			if (alphaAcc > 0.995f){
				break;
			}

		}

		if (sum.x==0.0f && sum.y==0.0f && sum.z==0.0f && sum.w==0.0f){
			sum = f4ColorBG;
		}

		unsigned int result = rgbaFloatToInt(sum);

		pPixelData[nIdx*3]	 = result & 0xFF; //R
		pPixelData[nIdx*3+1] = (result>>8) & 0xFF; //G
		pPixelData[nIdx*3+2] = (result>>16) & 0xFF; //B
	}
}

extern "C"
void cuda_render(unsigned char* pVR, int width, int height, float xTranslate, float yTranslate, float scale, bool invertZ, RGBA colorBG)
{
	if (width>width_vr || height>height_vr)
	{
		if (p_vr != 0)
			checkCudaErrors(cudaFree(p_vr));
		width_vr = width;
		height_vr = height;
		checkCudaErrors(cudaMalloc( (void**)&p_vr, width_vr * height_vr * 3 * sizeof(unsigned char) ));
	}

	dim3 blockSize(32, 32);
	dim3 gridSize( (width-1)/blockSize.x+1, (height-1)/blockSize.y+1 );

	float4 clrBG = make_float4(colorBG.red, colorBG.green, colorBG.blue, colorBG.alpha);

	d_render<<<gridSize, blockSize>>>(
		p_vr,
		volume_text_obj,
		maskText,
		width,
		height,
		xTranslate,
		yTranslate,
		scale,
		max_per,
		spacing,
		normal,
		voi,
		volume_size,
		invertZ,
		clrBG
	);
	cudaError_t t = cudaMemcpy( pVR, p_vr, width*height*3*sizeof(unsigned char), cudaMemcpyDeviceToHost );
}
