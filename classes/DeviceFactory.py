

from classes.Device import Device, CloudServer


class DeviceFactory:
    """
    Class to define a factory of devices, characterized by a key that 
    identifies the device type

    Args:
        devices (dict {str: Device}): Dictionary of devices
    """

    def __init__(self):

        self.devices = {}

    def register(self, key, device):
        """
        Method to register a new device to the factory
        Args:
            key (str): device type
            device (Device): device to be added
        """
        self.devices[key] = device

    def initialize(self, key, **kwargs):
        """
        Method to initialize a new device from the factory

        Args:
            key (str): device type
            **kwargs : Keyword arguments required by the device constructor
        """
        device = self.devices.get(key)
        if not device:
            raise ValueError(key)
        return device(**kwargs)


# initialization of the factory
Devicef = DeviceFactory()
Devicef.register("SEW", Device)
Devicef.register("Phone", Device)
Devicef.register("Cloud", CloudServer)
