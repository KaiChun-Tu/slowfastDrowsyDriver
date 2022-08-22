import math
import utils

def get_num(point_dict, name, axis):
	num = point_dict.get(f'{name}')[axis]
	num = float(num)
	return num

def cross_point(line1, line2):  
	x1 = line1[0]  
	y1 = line1[1]
	x2 = line1[2]
	y2 = line1[3]

	x3 = line2[0]
	y3 = line2[1]
	x4 = line2[2]
	y4 = line2[3]

	k1 = (y2 - y1) * 1.0 / (x2 - x1) 
	b1 = y1 * 1.0 - x1 * k1 * 1.0  
	if (x4 - x3) == 0: 
		k2 = None
		b2 = 0
	else:
		k2 = (y4 - y3) * 1.0 / (x4 - x3)
		b2 = y3 * 1.0 - x3 * k2 * 1.0
	if k2 == None:
		x = x3
	else:
		x = (b2 - b1) * 1.0 / (k1 - k2)
	y = k1 * x * 1.0 + b1 * 1.0
	return [x, y]

def point_line(point,line):
	x1 = line[0]  
	y1 = line[1]
	x2 = line[2]
	y2 = line[3]

	x3 = point[0]
	y3 = point[1]

	k1 = (y2 - y1)*1.0 /(x2 -x1) 
	b1 = y1 *1.0 - x1 *k1 *1.0
	k2 = -1.0/k1
	b2 = y3 *1.0 -x3 * k2 *1.0
	x = (b2 - b1) * 1.0 /(k1 - k2)
	y = k1 * x *1.0 +b1 *1.0
	return [x,y]

def point_point(point_1,point_2):
	x1 = point_1[0]
	y1 = point_1[1]
	x2 = point_2[0]
	y2 = point_2[1]
	distance = ((x1-x2)**2 +(y1-y2)**2)**0.5
	return distance

def find_pose(point_dict):
	#yaw
		point1 = [get_num(point_dict, 1, 0), get_num(point_dict, 1, 1)]
		point31 = [get_num(point_dict, 31, 0), get_num(point_dict, 31, 1)]
		point51 = [get_num(point_dict, 51, 0), get_num(point_dict, 51, 1)]
		crossover51 = point_line(point51, [point1[0], point1[1], point31[0], point31[1]])
		yaw_mean = point_point(point1, point31) / 2
		yaw_right = point_point(point1, crossover51)
		yaw = (yaw_mean - yaw_right) / yaw_mean
		yaw = int(yaw * 71.58 + 0.7037) + utils.VAR_yaw

		#pitch
		pitch_dis = point_point(point51, crossover51)
		if point51[1] < crossover51[1]:
			pitch_dis = -pitch_dis
		pitch = int(1.497 * pitch_dis + 18.97) + utils.VAR_pitch

		#roll
		roll_tan = abs(get_num(point_dict,60,1) - get_num(point_dict,72,1)) / abs(get_num(point_dict,60,0) - get_num(point_dict,72,0))
		roll = math.atan(roll_tan)
		roll = math.degrees(roll)
		if get_num(point_dict, 60, 1) > get_num(point_dict, 72, 1):
			roll = -roll
		roll = int(roll) + utils.VAR_roll

		return yaw, pitch, roll
