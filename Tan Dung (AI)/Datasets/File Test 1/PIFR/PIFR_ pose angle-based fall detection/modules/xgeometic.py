import pandas as pd
import numpy as np
import math

def extract_features(df: pd.DataFrame, extract_feat_names:list):
    if 'center_mass_x' in extract_feat_names: df['center_mass_x'], df['center_mass_y'] = zip(*df.apply(lambda row: center_of_mass(row, df.columns.tolist()), axis=1))
    if 'shoulder_nose_angle' in extract_feat_names: df['shoulder_nose_angle'] = df.apply(lambda row: calculate_angle_shoulder_nose(row), axis=1)
    if 'torso_angle' in extract_feat_names:df['torso_angle'] = df.apply(compute_torso_angle, axis=1)
    if 'width_height_ratio' in extract_feat_names: df['width_height_ratio'] = df.apply(width_height_ratio, axis=1)
    if 'hip_angle' in extract_feat_names: df['hip_angle'] = df.apply(compute_hip_angle, axis=1)
    if 'shoulder_angle' in extract_feat_names: df['shoulder_angle'] = df.apply(compute_shoulder_angle, axis=1)
    if 'left_leg_angle' in extract_feat_names: df['left_leg_angle'], df['right_leg_angle'] = zip(*df.apply(compute_leg_angles, axis=1))
    if 'nose_to_ankle_angle'in extract_feat_names: df['nose_to_ankle_angle'] = df.apply(compute_nose_to_ankle_angle, axis=1)
    df = df[extract_feat_names]
    return df

def width_height_ratio(row):
    _, _, width, height = row['xywh']
    return width/height

def center_of_mass(row, cols_name):
    xs, ys = [], [] 
    list_of_tuples = [(cols_name[i], cols_name[i+1]) for i in range(0, len(cols_name)-1, 2)]
    for (x, y) in list_of_tuples:
        xs.append(row[x])
        ys.append(row[y])
    total_x = sum(xs)
    total_y = sum(ys)
    
    n = len(list_of_tuples)
    
    C_x = total_x / n
    C_y = total_y / n
    
    return C_x, C_y

def calculate_angle_shoulder_nose(row):
    """Calculate the angle between the points A, B, and C."""
    A, B, C = [row['left_shoulder_x'], row['left_shoulder_y']], [row['nose_x'], row['nose_y']], [row['right_shoulder_x'], row['right_shoulder_y']],
    # Create vectors BA and BC
    BA = [A[0]-B[0], A[1]-B[1]]
    BC = [C[0]-B[0], C[1]-B[1]]

    # Calculate the dot product
    dot_product = BA[0]*BC[0] + BA[1]*BC[1]

    # Calculate the magnitude (norm) of both vectors
    norm_BA = math.sqrt(BA[0]**2 + BA[1]**2)
    norm_BC = math.sqrt(BC[0]**2 + BC[1]**2)

    # Calculate the angle using the arccosine of the dot product divided by the product of the norms
    angle = math.acos(dot_product / (norm_BA * norm_BC))

    # Convert the angle from radians to degrees
    angle_deg = math.degrees(angle)

    return angle_deg

# Modifying the function to accept a row
def compute_torso_angle(row):
    # Calculate midpoint of the hips
    mid_hip_x = (row['left_hip_x'] + row['right_hip_x']) / 2
    mid_hip_y = (row['left_hip_y'] + row['right_hip_y']) / 2
    
    # Create the vector for the torso
    torso_vector = [mid_hip_x - row['nose_x'], mid_hip_y - row['nose_y']]
    
    # Vertical vector (assuming the coordinate system has y increasing upwards)
    vertical_vector = [0, 1]  
    
    # Calculate dot product
    dot_product = torso_vector[1]  # because x-component of vertical_vector is 0
    
    # Calculate magnitudes
    magnitude_torso = math.sqrt(torso_vector[0]**2 + torso_vector[1]**2)
    
    # Calculate cosine of angle
    cos_angle = dot_product / (magnitude_torso * 1)  # multiplied by 1 because magnitude of vertical_vector is 1
    
    # Compute the angle in radians
    angle_radians = math.acos(cos_angle)
    
    # Convert the angle to degrees
    angle = math.degrees(angle_radians)
    return angle

def compute_hip_angle(row):
    # Create the vector for the hips
    hip_vector = [row['right_hip_x'] - row['left_hip_x'], row['right_hip_y'] - row['left_hip_y']]
    
    # Horizontal vector (assuming the coordinate system has x increasing to the right)
    horizontal_vector = [1, 0]
    
    # Calculate dot product
    dot_product = hip_vector[0]  # because y-component of horizontal_vector is 0
    
    # Calculate magnitudes
    magnitude_hip = math.sqrt(hip_vector[0]**2 + hip_vector[1]**2)
    
    # Calculate cosine of angle
    cos_angle = dot_product / (magnitude_hip * 1)  # multiplied by 1 because magnitude of horizontal_vector is 1
    
    # Compute the angle in radians
    angle_radians = math.acos(cos_angle)
    
    # Convert the angle to degrees
    angle_degrees = math.degrees(angle_radians)
    
    return angle_degrees

def compute_shoulder_angle(row):
    # Create the vector for the shoulders
    shoulder_vector = [row['right_shoulder_x'] - row['left_shoulder_x'], 
                       row['right_shoulder_y'] - row['left_shoulder_y']]
    
    # Horizontal vector (assuming the coordinate system has x increasing to the right)
    horizontal_vector = [1, 0]
    
    # Calculate dot product
    dot_product = shoulder_vector[0]  # because y-component of horizontal_vector is 0
    
    # Calculate magnitudes
    magnitude_shoulder = math.sqrt(shoulder_vector[0]**2 + shoulder_vector[1]**2)
    
    # Calculate cosine of angle
    cos_angle = dot_product / (magnitude_shoulder * 1)  # multiplied by 1 because magnitude of horizontal_vector is 1
    
    # Compute the angle in radians
    angle_radians = math.acos(cos_angle)
    
    # Convert the angle to degrees
    angle_degrees = math.degrees(angle_radians)
    
    return angle_degrees

def compute_leg_angles(row):
    # Vectors for left and right legs
    left_leg_upper = [row['left_knee_x'] - row['left_hip_x'], row['left_knee_y'] - row['left_hip_y']]
    left_leg_lower = [row['left_ankle_x'] - row['left_knee_x'], row['left_ankle_y'] - row['left_knee_y']]
    
    right_leg_upper = [row['right_knee_x'] - row['right_hip_x'], row['right_knee_y'] - row['right_hip_y']]
    right_leg_lower = [row['right_ankle_x'] - row['right_knee_x'], row['right_ankle_y'] - row['right_knee_y']]
    
    # Calculate dot products for left and right leg
    dot_product_left = sum(a*b for a, b in zip(left_leg_upper, left_leg_lower))
    dot_product_right = sum(a*b for a, b in zip(right_leg_upper, right_leg_lower))
    
    # Calculate magnitudes for left and right leg vectors
    magnitude_left_upper = math.sqrt(sum(val**2 for val in left_leg_upper))
    magnitude_left_lower = math.sqrt(sum(val**2 for val in left_leg_lower))
    
    magnitude_right_upper = math.sqrt(sum(val**2 for val in right_leg_upper))
    magnitude_right_lower = math.sqrt(sum(val**2 for val in right_leg_lower))
    
    # Calculate cosine of angles for left and right leg
    cos_angle_left = dot_product_left / (magnitude_left_upper * magnitude_left_lower)
    cos_angle_right = dot_product_right / (magnitude_right_upper * magnitude_right_lower)
    
    # Compute the angles in radians
    angle_left_radians = math.acos(cos_angle_left)
    angle_right_radians = math.acos(cos_angle_right)
    
    # Convert the angles to degrees
    angle_left_degrees = math.degrees(angle_left_radians)
    angle_right_degrees = math.degrees(angle_right_radians)
    
    return angle_left_degrees, angle_right_degrees

def compute_nose_to_ankle_angle(row):
    # Calculate midpoint of the ankles
    mid_ankle_x = (row['left_ankle_x'] + row['right_ankle_x']) / 2
    mid_ankle_y = (row['left_ankle_y'] + row['right_ankle_y']) / 2
    
    # Create the vector from nose to ankle midpoint
    vector = [mid_ankle_x - row['nose_x'], mid_ankle_y - row['nose_y']]
    
    # Vertical vector (assuming the coordinate system has y increasing upwards)
    vertical_vector = [0, 1]  
    
    # Calculate dot product
    dot_product = vector[1]  # because x-component of vertical_vector is 0
    
    # Calculate magnitudes
    magnitude_vector = math.sqrt(vector[0]**2 + vector[1]**2)
    
    # Calculate cosine of angle
    cos_angle = dot_product / magnitude_vector  # multiplied by 1 because magnitude of vertical_vector is 1
    
    # Compute the angle in radians
    angle_radians = math.acos(cos_angle)
    
    # Convert the angle to degrees
    angle_degrees = math.degrees(angle_radians)
    
    return angle_degrees