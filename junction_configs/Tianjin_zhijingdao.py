'''
@Author: WANG Maonan
@Date: 2026-06-01 01:11:04
@Description: 天津知景道路口配置
@LastEditTime: 2026-06-25 23:19:40
@LastEditors: WANG Maonan
'''
JUNCTION = {
    "tls_id": "INT1",
    # ===== easy 路网 =====
    "easy_low_density": {
        "num_phases": 4, 
        "num_seconds": 1000,
        "fix_phase_durations": [3, 3, 3, 3],
    },
    "easy_high_density": {
        "num_phases": 4,
        "num_seconds": 1000,
        "fix_phase_durations": [3, 3, 3, 3],
    },
    "easy_fluctuating_commuter": {
        "num_phases": 4,
        "num_seconds": 1000,
        "fix_phase_durations": [3, 3, 3, 3],
    },
    "easy_increasing_demand": {
        "num_phases": 4,
        "num_seconds": 1000,
        "fix_phase_durations": [3, 3, 3, 3],
    },
    "easy_random_perturbation": {
        "num_phases": 4,
        "num_seconds": 1000,
        "fix_phase_durations": [3, 3, 3, 3],
    },
    # ===== normal 路网 =====
    "normal_low_density": {
        "num_phases": 4, 
        "num_seconds": 1000,
        "fix_phase_durations": [3, 3, 3, 3],
    },
    "normal_high_density": {
        "num_phases": 4,
        "num_seconds": 1000,
        "fix_phase_durations": [3, 3, 3, 3],
    },
    "normal_fluctuating_commuter": {
        "num_phases": 4,
        "num_seconds": 1000,
        "fix_phase_durations": [3, 3, 3, 3],
    },
    "normal_increasing_demand": {
        "num_phases": 4,
        "num_seconds": 1000,
        "fix_phase_durations": [3, 3, 3, 3],
    },
    "normal_random_perturbation": {
        "num_phases": 4,
        "num_seconds": 1000,
        "fix_phase_durations": [3, 3, 3, 3],
    },
}

EVENTS = {
    "event_1": {
        "accidents": [
            {
                "id": "accident_01",
                "depart_time": 100,
                "edge_id": "417937574#1.74",
                "lane_index": 1,
                "position": 203,
                "type": "barrier",
                "duration": 70,
            },
            {
                "id": "accident_02",
                "depart_time": 100,
                "edge_id": "417937574#1.74",
                "lane_index": 2,
                "position": 203,
                "type": "barrier",
                "duration": 70,
            },
        ],
        "special_vehicles": [
            {
                "id": "police_01",
                "type": "police",
                "depart_time": 75,
                "route": ["339537367#2.7", "339537367#4"],
            },
            {
                "id": "ambulance_02",
                "type": "emergency",
                "depart_time": 250,
                "route": ["339537541#3", "339537541#5"],
            },
        ],
    },
}