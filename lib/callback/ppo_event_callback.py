"""
callback_ppo
ppo를 위한 callback 클래스를 정의합니다.

"""
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class PPOEventCallback(BaseCallback):
    """
    --- 적용 방법 ---
    callback = PPOEventCallback(check_freq=1000, save_path="./logs/", verbose=2)
    model.learn(total_timesteps=20000, callback=callback)
    """
    def __init__(self, verbose=1, check_freq: int=100, is_save_model=False, save_path: str=None, cb_verbose=1):
        super(PPOEventCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self._is_save_mode = is_save_model
        self._cb_verbose = cb_verbose
        self.save_path = save_path
        self.best_mean_reward = -np.inf

    def _on_training_start(self) -> None:
        """학습이 시작 이벤트"""
        if self.verbose > 0:
            print("🚀 [Event] 학습을 시작합니다. 하이퍼파라미터 초기화 중...")

    def _on_rollout_start(self) -> None:
        """새로운 데이터 수집 세션이 시작"""
        if self.verbose > 1:
            print(f"\n📥 [Event] 데이터 수집 시작 (현재 스텝: {self.num_timesteps})")

    # def _on_step(self, locals, globals) -> bool:
    def _on_step(self, locals, globals) -> bool:
        """새로운 스텝"""
        # # 주기적으로 성능 체크 및 모델 저장
        # if self.n_calls % self.check_freq == 0:
        #     if len(self.model.ep_info_buffer) > 0:
        #         mean_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
                
        #         if mean_reward > self.best_mean_reward:
        #             self.best_mean_reward = mean_reward
        #             if self._is_save_mode:
        #                 self.model.save(f"{self.save_path}/best_model")
        #             if self.verbose > 2:
        #                 print(f"✅ [Update] 최고 보상 갱신: {mean_reward:.2f} -> 모델 저장 완료")
        return True

    def _on_rollout_end(self) -> None:
        """데이터 수집이 끝나고 정책 업데이트(Update) 직전에 실행"""
        if self._cb_verbose > 1:
            print(f"📤 [Event] 데이터 수집 완료. 정책 업데이트를 시작합니다.")

    def _on_training_end(self) -> None:
        """학습 전체 과정이 종료될 때 실행"""
        if self.verbose > 0:
            print("🏁 [Event] 모든 학습 과정이 종료되었습니다.")
            print(f"최종 최고 평균 보상: {self.best_mean_reward:.2f}")
