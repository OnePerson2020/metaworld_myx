"""Rotation utilities for quaternion and Euler angle conversions."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def quat2euler(quat: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Convert quaternion to Euler angles (roll, pitch, yaw).
    
    Args:
        quat: Quaternion in [w, x, y, z] format
        
    Returns:
        Euler angles in [roll, pitch, yaw] format (radians)
    """
    w, x, y, z = quat
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if np.abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.array([roll, pitch, yaw])


def euler2quat(euler: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Convert Euler angles to quaternion.
    
    Args:
        euler: Euler angles in [roll, pitch, yaw] format (radians)
        
    Returns:
        Quaternion in [w, x, y, z] format
    """
    roll, pitch, yaw = euler
    
    # Abbreviations for the various angular functions
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return np.array([w, x, y, z])


def normalize_quat(quat: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Normalize a quaternion to unit length.
    
    Args:
        quat: Quaternion in [w, x, y, z] format
        
    Returns:
        Normalized quaternion
    """
    norm = np.linalg.norm(quat)
    if norm == 0:
        return np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
    return quat / norm


def quat_multiply(q1: npt.NDArray[np.float64], q2: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Multiply two quaternions.
    
    Args:
        q1: First quaternion in [w, x, y, z] format
        q2: Second quaternion in [w, x, y, z] format
        
    Returns:
        Product quaternion in [w, x, y, z] format
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return np.array([w, x, y, z])
