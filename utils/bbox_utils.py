def get_foot_position(bbox):
    return int((bbox[0]+bbox[2])//2), bbox[3]

def get_box_center(bbox):
    return int((bbox[0]+bbox[2])//2), int((bbox[1]+bbox[3])//2)

def calculate_distance(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5