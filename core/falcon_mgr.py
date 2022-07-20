from threading import Lock

from core.falcon import Falcon


__all__ = ('falcon_mgr',)


class FalconMgr:
    def __int__(self):
        self.__uid_to_falcons = {}
        self.__instance_lock = Lock()

    def get_falcon(self, falcon_uid):
        falcon = self.__uid_to_falcons.get(falcon_uid)
        if not falcon:
            with self.__instance_lock:
                if falcon_uid not in self.__uid_to_falcons:
                    self.__uid_to_falcons[falcon_uid] = Falcon()

        return falcon


falcon_mgr = FalconMgr()
