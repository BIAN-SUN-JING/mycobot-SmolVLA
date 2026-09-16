import time
from dataclasses import dataclass, field
from typing import Any
import serial.tools.list_ports
from pymycobot import MyCobot280

from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots import Robot, RobotConfig


@RobotConfig.register_subclass("mycobot280")
@dataclass(kw_only=True)
class MyCobot280RobotConfig(RobotConfig):
    port: str = "auto"
    baudrate: int = 1_000_000
    motion_speed: int = 5
    max_delta_deg: float = 1.0
    # 现在禁止运动。
    enable_motion: bool = False
    JOINT_LIMITS = {
        "joint_1": (-168.0, 168.0),
        "joint_2": (-140.0, 140.0),
        "joint_3": (-150.0, 150.0),
        "joint_4": (-150.0, 150.0),
        "joint_5": (-155.0, 160.0),
        "joint_6": (-180.0, 180.0),
    }

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "camera1": OpenCVCameraConfig(
                index_or_path=1,
                fps=30,
                width=640,
                height=480,
            )
        }
    )


class MyCobot280Robot(Robot):
    config_class = MyCobot280RobotConfig
    name = "mycobot280"

    @staticmethod
    def find_mycobot_port() -> str:
        ports = list(serial.tools.list_ports.comports())

        matches = []

        for port in ports:
            if (
                    port.vid == 0x1A86
                    and port.pid == 0x7523
            ):
                matches.append(port.device)

        if len(matches) == 0:
            raise ConnectionError(
                "CH340 serial device not found. "
                "Please check USB cable and robot power."
            )

        if len(matches) > 1:
            raise ConnectionError(
                f"Multiple CH340 devices found: {matches}. "
                "Please specify the port manually."
            )

        return matches[0]
    JOINT_NAMES = [
        "joint_1",
        "joint_2",
        "joint_3",
        "joint_4",
        "joint_5",
        "joint_6",
    ]
    JOINT_LIMITS = {
        "joint_1": (-168.0, 168.0),
        "joint_2": (-140.0, 140.0),
        "joint_3": (-150.0, 150.0),
        "joint_4": (-150.0, 150.0),
        "joint_5": (-155.0, 160.0),
        "joint_6": (-180.0, 180.0),
    }

    def __init__(self, config: MyCobot280RobotConfig):
        super().__init__(config)

        self.config = config

        self.mc: MyCobot280 | None = None

        self.cameras = make_cameras_from_configs(
            config.cameras
        )

        self._connected = False

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {
            f"{joint}.pos": float
            for joint in self.JOINT_NAMES
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple[int, int, int]]:
        return {
            name: (
                camera_config.height,
                camera_config.width,
                3,
            )
            for name, camera_config in self.config.cameras.items()
        }

    @property
    def observation_features(self) -> dict:
        return {
            **self._motors_ft,
            **self._cameras_ft,
        }

    @property
    def action_features(self) -> dict:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        if not self._connected:
            return False

        if self.mc is None:
            return False

        return all(
            camera.is_connected
            for camera in self.cameras.values()
        )

    @property
    def is_calibrated(self) -> bool:
        # myCobot 自身负责关节编码器/零位，
        # 第一版 LeRobot adapter 不额外做 calibration。
        return True

    def connect(self, calibrate: bool = True) -> None:
        if self._connected:
            return

        if self.config.port.lower() == "auto":
            port = self.find_mycobot_port()
        else:
            port = self.config.port

        print(
            f"[myCobot] Connecting to "
            f"{port} @ {self.config.baudrate}..."
        )

        self.mc = MyCobot280(
            port,
            self.config.baudrate,
        )

        time.sleep(2.0)

        connected = self.mc.is_controller_connected()

        if connected != 1:
            try:
                self.mc.close()
            except Exception:
                pass

            self.mc = None

            raise ConnectionError(
                "myCobot controller communication failed. "
                f"is_controller_connected()={connected}"
            )

        print("[myCobot] Controller connected.")

        try:
            for name, camera in self.cameras.items():
                print(f"[Camera] Connecting {name}...")
                camera.connect()

            self._connected = True

            print("[LeRobot] Robot connected.")

        except Exception:
            try:
                self.mc.close()
            except Exception:
                pass

            self.mc = None

            for camera in self.cameras.values():
                try:
                    camera.disconnect()
                except Exception:
                    pass

            raise

    def configure(self) -> None:
        # 目前不修改机械臂控制模式。
        pass

    def calibrate(self) -> None:
        # 当前阶段无额外 LeRobot calibration。
        pass

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise ConnectionError(
                "myCobot is not connected."
            )

        if self.mc is None:
            raise RuntimeError(
                "myCobot controller object is missing."
            )

        angles = self.mc.get_angles()

        if angles is None or len(angles) != 6:
            raise RuntimeError(
                f"Invalid joint angles returned: {angles}"
            )

        observation: dict[str, Any] = {}

        for joint_name, angle in zip(
            self.JOINT_NAMES,
            angles,
            strict=True,
        ):
            observation[f"{joint_name}.pos"] = float(angle)

        for camera_name, camera in self.cameras.items():
            observation[camera_name] = camera.async_read(
                timeout_ms=1000
            )

        return observation

    def send_action(
            self,
            action: dict[str, Any],
    ) -> dict[str, Any]:

        if not self.config.enable_motion:
            raise RuntimeError(
                "Robot motion is DISABLED."
            )

        if self.mc is None:
            raise RuntimeError(
                "myCobot controller is not connected."
            )

        current = self.mc.get_angles()

        if current is None or len(current) != 6:
            raise RuntimeError(
                f"Invalid current angles: {current}"
            )

        target = []

        for i, joint_name in enumerate(
                self.JOINT_NAMES
        ):
            key = f"{joint_name}.pos"

            if key not in action:
                raise KeyError(
                    f"Missing action key: {key}"
                )

            requested = float(action[key])
            current_angle = float(current[i])

            # 单次最大变化限制
            delta = requested - current_angle

            if delta > self.config.max_delta_deg:
                requested = (
                        current_angle
                        + self.config.max_delta_deg
                )

            elif delta < -self.config.max_delta_deg:
                requested = (
                        current_angle
                        - self.config.max_delta_deg
                )

            low, high = self.JOINT_LIMITS[
                joint_name
            ]

            requested = max(
                low,
                min(high, requested),
            )

            target.append(requested)

        print(
            "[Safety] current:",
            [round(v, 2) for v in current],
        )

        print(
            "[Safety] target:",
            [round(v, 2) for v in target],
        )

        executed = {}

        for i, joint_name in enumerate(
            self.JOINT_NAMES
        ):
            current_angle = float(current[i])
            target_angle = float(target[i])

            delta = target_angle - current_angle

            # 非常小的变化不执行
            if abs(delta) < 0.2:
                executed[
                    f"{joint_name}.pos"
                ] = current_angle
                continue

            joint_id = i + 1

            print(
                f"[Motion] J{joint_id}: "
                f"{current_angle:.2f} -> "
                f"{target_angle:.2f}"
            )

            self.mc.send_angle(
                joint_id,
                target_angle,
                self.config.motion_speed,
            )

            executed[
                f"{joint_name}.pos"
            ] = target_angle

        return executed

    def disconnect(self) -> None:
        for camera in self.cameras.values():
            try:
                if camera.is_connected:
                    camera.disconnect()
            except Exception as exc:
                print(
                    f"[Camera] Disconnect warning: {exc}"
                )

        if self.mc is not None:
            try:
                self.mc.close()
            except Exception as exc:
                print(
                    f"[myCobot] Serial close warning: {exc}"
                )

        self.mc = None
        self._connected = False

        print("[LeRobot] Robot disconnected.")