import torch
import torch.nn as nn


class Module(nn.Module):
    def to(self, *args, **kwargs):
        """
        Override the default `to` method to move all tensor attributes and
        submodules to the specified device.
        """
        # Move the module itself
        module = super(Module, self).to(*args, **kwargs)

        device = kwargs.get('device', args[0] if len(args) > 0 else None)
        if device is None:
            raise ValueError("Device not specified in `to` method.")

        # Helper function to move attributes to device
        def move_to_device(value, device):
            if isinstance(value, torch.Tensor):
                return value.to(device)
            elif isinstance(value, nn.Parameter):
                return nn.Parameter(value.to(device))
            elif isinstance(
                    value,
                    (nn.Module, nn.LayerNorm, nn.ModuleList)
            ):
                return value.to(device)
            elif isinstance(value, nn.Linear):
                value.weight = value.weight.to(device)
                if value.bias is not None:
                    value.bias = value.bias.to(device)
                return value
            elif isinstance(value, Module):
                return value.to(device)
            else:
                return value

        # Move all tensor attributes and submodules to the specified device
        for attr, value in self.__dict__.items():
            moved_value = move_to_device(value, device)
            object.__setattr__(module, attr, moved_value)

        return module