# ---------------------------------------------------------
# 콜백 정의
# 학습에 상태에 대한 모니터링을 위한 콜백 코드 정의
# ---------------------------------------------------------
from stable_baselines3.common.callbacks import BaseCallback
import time
from datetime import datetime

class CustomCallback(BaseCallback):
    """
    A custom callback that derives from BaseCallback.
    """
    def __init__(self, verbose=0, cb_verbose=0, save_mode=False, verbose_interval=10):
        super(CustomCallback, self).__init__(verbose)
        self.rollout_count = 0
        self._save_mode = save_mode
        self._cb_verbose = cb_verbose
        self._verb_interval = verbose_interval

    def _on_step(self) -> bool:
        # This method is called at every step
        # Access self.model (the RecurrentPPO instance)
        # Access local/global variables
        return True

    def _on_rollout_end(self) -> None:
        """
        데이터 수집(Rollout)이 한 번 완료될 때마다 호출됩니다.
        네트워크 업데이트(Optimization) 직전 시점입니다.
        """
        # 롤아웃이 끝날 때마다 현재까지의 평균 보상 등을 로깅하거나 모델을 임시 저장할 수 있습니다.
        if self._cb_verbose > 0:
            if self.rollout_count % self._verb_interval == 0:
                print(f"--- [{self.rollout_count}번째 롤아웃 종료] 모델 업데이트를 시작합니다.[{datetime.now()}]")

        # 예: 특정 주기마다 모델 저장
        if self._save_mode:
            if self.rollout_count % 5 == 0:
                self.model.save(f"recurrent_ppo_checkpoint_{self.rollout_count}")

    def _on_training_end(self) -> None:
        """
        model.learn()의 모든 과정이 끝났을 때 1회 호출됩니다.
        """
        print(f"학습이 모두 완료되었습니다! [{datetime.now()}")
        if self._save_mode:
            # 최종 모델 저장 또는 학습 결과 요약 출력
            self.model.save("recurrent_ppo_final_model")

    def _on_training_start(self) -> None:
        """
        model.learn()이 호출된 직후, 학습 루프가 시작되기 전 1회 실행됩니다.
        """
        self.start_time = time.time()
        if self._cb_verbose:        
            print(">>> [학습 시작] 환경 및 모델 초기화가 완료되었습니다.")
        # 예: 텐서보드에 초기 하이퍼파라미터 기록 등

    def _on_rollout_start(self) -> None:
        """
        새로운 데이터를 수집(Rollout)하기 위해 환경과 상호작용을 시작할 때마다 호출됩니다.
        """
        self.rollout_count += 1
        if self._cb_verbose:
            if self.rollout_count % self._verb_interval == 0:
                print(f"\n--- [롤아웃 #{self.rollout_count} 시작] 데이터를 수집합니다. ---[{datetime.now()}]")
        # 예: LSTM의 초기 Hidden State 확인이나 특정 환경 변수 리셋 모니터링
