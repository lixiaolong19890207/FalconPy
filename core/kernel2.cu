#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_math.h>
#include "Defines.h"

using namespace MonkeyGL;

typedef struct {
	float3 m[3];
} float3x3;

cudaTextureObject_t volumeText;
cudaArray* d_volumeArray = 0;

__constant__ cudaTextureObject_t constTransferFuncTexts[MAXOBJECTCOUNT+1];
cudaTextureObject_t transferFuncTexts[MAXOBJECTCOUNT+1];
cudaArray *d_transferFuncArrays[MAXOBJECTCOUNT+1];

__constant__ float3 constAlphaAndWWWL[MAXOBJECTCOUNT+1];
float3 alphaAndWWWL[MAXOBJECTCOUNT+1];

cudaTextureObject_t maskText;
cudaArray* d_maskArray = 0;

float3 m_f3Nor, m_f3Spacing, m_f3maxper;
VOI m_voi;
cudaExtent m_volumeSize;

__constant__ float3x3 constTransposeTransformMatrix;
__constant__ float3x3 constTransformMatrix;

unsigned char* d_pVR = 0;
int nWidth_VR = 0;
int nHeight_VR = 0;
short* d_pMPR = 0;
int nWidth_MPR = 0;
int nHeight_MPR = 0;

extern "C"
void cu_copyVolumeData( short* h_volumeData, cudaExtent volumeSize)
{
	m_volumeSize = make_cudaExtent(volumeSize.width, volumeSize.height, volumeSize.depth);

	if (d_volumeArray != 0)
	{
		checkCudaErrors(cudaFreeArray(d_volumeArray));
		d_volumeArray = 0;
		volumeText = 0;
	}
	if (d_maskArray != 0)
	{
		checkCudaErrors(cudaFreeArray(d_maskArray));
		d_maskArray = 0;
		maskText = 0;
	}

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<short>();
	checkCudaErrors( cudaMalloc3DArray(&d_volumeArray, &channelDesc, m_volumeSize) );

	cudaMemcpy3DParms copyParams = {0};
	copyParams.dstArray = d_volumeArray;
	copyParams.extent   = m_volumeSize;
	copyParams.kind     = cudaMemcpyHostToDevice;
	copyParams.srcPtr   = make_cudaPitchedPtr(
		(void*)h_volumeData,
		m_volumeSize.width*sizeof(short),
		m_volumeSize.width,
		m_volumeSize.height
	);

	checkCudaErrors( cudaMemcpy3D(&copyParams) );

	cudaResourceDesc texRes;
	memset(&texRes, 0, sizeof(cudaResourceDesc));

	texRes.resType = cudaResourceTypeArray;
	texRes.res.array.array = d_volumeArray;

	cudaTextureDesc texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));

	texDescr.normalizedCoords = true;  // access with normalized texture coordinates
	texDescr.filterMode = cudaFilterModeLinear;  // linear interpolation

	texDescr.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
	texDescr.addressMode[1] = cudaAddressModeClamp;
	texDescr.addressMode[2] = cudaAddressModeClamp;

	texDescr.readMode = cudaReadModeNormalizedFloat;

	checkCudaErrors( cudaCreateTextureObject(&volumeText, &texRes, &texDescr, NULL) );
}

extern "C"
void cu_InitCommon(float fxSpacing, float fySpacing, float fzSpacing)
{
	d_pVR = 0;
	nWidth_VR = 0;
	nHeight_VR = 0;

	m_f3Spacing.x = fxSpacing;
	m_f3Spacing.y = fySpacing;
	m_f3Spacing.z = fzSpacing;
	m_f3Nor.x = 1.0f / m_volumeSize.width;
	m_f3Nor.y = 1.0f / m_volumeSize.height;
	m_f3Nor.z = 1.0f / m_volumeSize.depth;
	float fMaxSpacing = max(fxSpacing, max(fySpacing, fzSpacing));

	float fMaxLen = max(m_volumeSize.width*fxSpacing, max(m_volumeSize.height*fySpacing, m_volumeSize.depth*fzSpacing));
	m_f3maxper.x = 1.0f*fMaxLen/(m_volumeSize.width*fxSpacing);
	m_f3maxper.y = 1.0f*fMaxLen/(m_volumeSize.height*fySpacing);
	m_f3maxper.z = 1.0f*fMaxLen/(m_volumeSize.depth*fzSpacing);

    for (int i=0; i<MAXOBJECTCOUNT; i++){
        d_transferFuncArrays[i] = 0;
    }
}

extern "C"
bool cu_setTransferFunc( float* pTransferFunc, int nLenTransferFunc, unsigned char nLabel)
{
	if (nLabel >= MAXOBJECTCOUNT){
		return false;
	}

	cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));
    texRes.resType = cudaResourceTypeArray;

    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));
    texDescr.normalizedCoords = true;
    texDescr.filterMode = cudaFilterModeLinear;
    texDescr.addressMode[0] = cudaAddressModeClamp;
    texDescr.readMode = cudaReadModeElementType;

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();

    if (d_transferFuncArrays[nLabel] != 0)
	{
		checkCudaErrors(cudaFreeArray(d_transferFuncArrays[nLabel]));
		d_transferFuncArrays[nLabel] = 0;
	}
    checkCudaErrors(cudaMallocArray( &d_transferFuncArrays[nLabel], &channelDesc, nLenTransferFunc, 1));
    checkCudaErrors(
        cudaMemcpy2DToArray(
            d_transferFuncArrays[nLabel],
            0,
            0,
            pTransferFunc,
            0,
            nLenTransferFunc*sizeof(float4),
            1,
            cudaMemcp1yHostToDevice
        )
    );

    texRes.res.array.array = d_transferFuncArrays[nLabel];

    cudaTextureObject_t text = 0;
    checkCudaErrors(
        cudaCreateTextureObject(&text, &texRes, &texDescr, NULL)
    );

    transferFuncTexts[nLabel] = text;
    cudaMemcpyToSymbol(constTransferFuncTexts, transferFuncTexts, sizeof(transferFuncTexts));

    return true;
}

extern "C"
void cu_copyOperatorMatrix( float *pTransformMatrix, float *pTransposeTransformMatrix)
{
	checkCudaErrors( cudaMemcpyToSymbol(constTransformMatrix, pTransformMatrix, sizeof(float3)*3) );
	checkCudaErrors( cudaMemcpyToSymbol(constTransposeTransformMatrix, pTransposeTransformMatrix, sizeof(float3)*3) );
}

extern "C"
void cu_copyAlphaAndWWWL(float *pAlphaAndWWWL)
{
	checkCudaErrors( cudaMemcpyToSymbol(constAlphaAndWWWL, pAlphaAndWWWL, sizeof(float3)*MAXOBJECTCOUNT+1) );
}

extern "C"
void cu_setVOI(VOI voi)
{
	m_voi.left = voi.left;
	m_voi.right = voi.right;
	m_voi.anterior = voi.anterior;
	m_voi.posterior = voi.posterior;
	m_voi.head = voi.head;
	m_voi.foot = voi.foot;
}

__device__ float3 mul(const float3x3 &M, const float3 &v)
{
	float3 r;
	r.x = dot(v, M.m[0]);
	r.y = dot(v, M.m[1]);
	r.z = dot(v, M.m[2]);
	return r;
}

__device__ uint rgbaFloatToInt(float4 rgba)
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
	cudaTextureObject_t maskText,
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
		dirLight = normalize(mul(constTransformMatrix, dirLight));

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
			pos = mul(constTransformMatrix, pos);

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
			if(maskText == 0){
				label = 0;
			}
			else {
				label = tex3D<unsigned char>(maskText, nxIdx, nyIdx, nzIdx);
			}
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
void cu_render(unsigned char* pVR, int width, int height, float xTranslate, float yTranslate, float scale, bool invertZ, RGBA colorBG)
{
	if (width>nWidth_VR || height>nHeight_VR)
	{
		if (d_pVR != 0)
			checkCudaErrors(cudaFree(d_pVR));
		nWidth_VR = width;
		nHeight_VR = height;
		checkCudaErrors(cudaMalloc( (void**)&d_pVR, nWidth_VR*nHeight_VR*3*sizeof(unsigned char) ));
	}

	dim3 blockSize(32, 32);
	dim3 gridSize( (width-1)/blockSize.x+1, (height-1)/blockSize.y+1 );

	float4 clrBG = make_float4(colorBG.red, colorBG.green, colorBG.blue, colorBG.alpha);

	d_render<<<gridSize, blockSize>>>(
		d_pVR,
		volumeText,
		maskText,
		width,
		height,
		xTranslate,
		yTranslate,
		scale,
		m_f3maxper,
		m_f3Spacing,
		m_f3Nor,
		m_voi,
		m_volumeSize,
		invertZ,
		clrBG
	);
	cudaError_t t = cudaMemcpy( pVR, d_pVR, width*height*3*sizeof(unsigned char), cudaMemcpyDeviceToHost );
}
