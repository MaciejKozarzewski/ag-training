import torch


class Loss:
    def __init__(self, policy_weight: float, value_weight: float, mlh_weight: float):
        self._policy_weight = policy_weight
        self._value_weight = value_weight
        self._mlh_weight = mlh_weight

    def __call__(self, policy_output: torch.Tensor, policy_target: torch.Tensor,
                 value_output: torch.Tensor, value_target: torch.Tensor) -> dict:
        batch_size = policy_output.shape[0]

        policy_output = torch.log_softmax(self._flatten(policy_output), dim=1)
        policy_target = self._flatten(policy_target)
        policy_loss = torch.nn.functional.kl_div(policy_output, policy_target, reduction='sum', log_target=False)

        value_output = torch.log_softmax(value_output, dim=1)
        value_loss = torch.nn.functional.kl_div(value_output, value_target, reduction='sum', log_target=False)

        result = {'policy_loss': self._policy_weight * policy_loss / batch_size,
                  'value_loss': self._value_weight * value_loss / batch_size}

        return result

    @staticmethod
    def _flatten(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
