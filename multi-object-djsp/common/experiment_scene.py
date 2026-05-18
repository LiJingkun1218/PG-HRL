# Orthogonal experiment data - dynamic job shop scenarios
orthogonal_scenarios = [
    {
        'scenario_id': 'L9-1',
        'parameters': {
            'machine_count': 5,
            'tightness_factor': 1.5,
            'process_time_range': (5, 25),
            'utilization': 0.65
        },
        'metadata': {
            'description': 'Small-scale urgent orders',
            'characteristics': ['High time pressure', 'Light system load', 'Easier scheduling'],
            'expected_challenge': 'On-time delivery',
            'difficulty_level': 'Easy'
        }
    },
    {
        'scenario_id': 'L9-2', 
        'parameters': {
            'machine_count': 5,
            'tightness_factor': 3,
            'process_time_range': (1, 50),
            'utilization': 0.8
        },
        'metadata': {
            'description': 'Small-scale mixed production',
            'characteristics': ['High task variability', 'Normal load', 'Comprehensive test'],
            'expected_challenge': 'Load balancing',
            'difficulty_level': 'Medium'
        }
    },
    {
        'scenario_id': 'L9-3',
        'parameters': {
            'machine_count': 5,
            'tightness_factor': 5,
            'process_time_range': (1, 100),
            'utilization': 0.95
        },
        'metadata': {
            'description': 'Small-scale bottleneck production',
            'characteristics': ['Extreme op time variance', 'Saturated load', 'Prone to blocking'],
            'expected_challenge': 'Avoid deadlock',
            'difficulty_level': 'Hard'
        }
    },
    {
        'scenario_id': 'L9-4',
        'parameters': {
            'machine_count': 10,
            'tightness_factor': 1.5, 
            'process_time_range': (1, 50),
            'utilization': 0.95
        },
        'metadata': {
            'description': 'Mid-scale high pressure',
            'characteristics': ['One of the harshest', 'Urgent orders', 'Mixed tasks', 'Very high load'],
            'expected_challenge': 'System stability',
            'difficulty_level': 'Very hard'
        }
    },
    {
        'scenario_id': 'L9-5',
        'parameters': {
            'machine_count': 10,
            'tightness_factor': 3,
            'process_time_range': (1, 100),
            'utilization': 0.65
        },
        'metadata': {
            'description': 'Mid-scale heterogeneous and relaxed',
            'characteristics': ['High task variability', 'Light load', 'Tests load balance'],
            'expected_challenge': 'Resource optimization',
            'difficulty_level': 'Medium'
        }
    },
    {
        'scenario_id': 'L9-6',
        'parameters': {
            'machine_count': 10,
            'tightness_factor': 5,
            'process_time_range': (5, 25),
            'utilization': 0.8
        },
        'metadata': {
            'description': 'Mid-scale stable production',
            'characteristics': ['Most typical and balanced', 'Algorithm benchmark'],
            'expected_challenge': 'Overall performance',
            'difficulty_level': 'Medium'
        }
    },
    {
        'scenario_id': 'L9-7',
        'parameters': {
            'machine_count': 15,
            'tightness_factor': 1.5,
            'process_time_range': (1, 100),
            'utilization': 0.8
        },
        'metadata': {
            'description': 'Large-scale urgent heterogeneous',
            'characteristics': ['Urgent orders at scale', 'Task variability challenge'],
            'expected_challenge': 'Response speed and coordination',
            'difficulty_level': 'Hard'
        }
    },
    {
        'scenario_id': 'L9-8',
        'parameters': {
            'machine_count': 15,
            'tightness_factor': 3,
            'process_time_range': (5, 25),
            'utilization': 0.95
        },
        'metadata': {
            'description': 'Large-scale high load',
            'characteristics': ['Uniform tasks, very high load', 'Tests throughput and stability'],
            'expected_challenge': 'Production efficiency',
            'difficulty_level': 'Hard'
        }
    },
    {
        'scenario_id': 'L9-9',
        'parameters': {
            'machine_count': 15,
            'tightness_factor': 5,
            'process_time_range': (1, 50),
            'utilization': 0.65
        },
        'metadata': {
            'description': 'Large-scale relaxed mixed',
            'characteristics': ['Large scale with low pressure', 'Wide optimization space'],
            'expected_challenge': 'Optimality pursuit',
            'difficulty_level': 'Easy'
        }
    }
]

# Helper: scenario lookup
def find_scenario(machine_count, tightness_factor, process_time_range, utilization):
    """Find scenario by parameter combination.
    Returns: scenario_id if found, else None.
    """
    for scenario in orthogonal_scenarios:
        params = scenario['parameters']
        if (params['machine_count'] == machine_count and
            params['tightness_factor'] == tightness_factor and
            params['process_time_range'] == process_time_range and
            params['utilization'] == utilization):
            return scenario['scenario_id']
    return None
def get_all_scenarios():
    """Return parameters for all scenarios.
    Returns a list of dicts with scenario_id and parameters.
    """
    all_scenarios = []
    for scenario in orthogonal_scenarios:
        scenario_info = {
            'scenario_id': scenario['scenario_id'],
            'parameters': scenario['parameters'].copy(),
            'metadata': scenario['metadata'].copy()
        }
        all_scenarios.append(scenario_info)
    return all_scenarios


def get_scenario_by_id(scenario_id):
    """Return scenario parameters by ID.
    Args:
        scenario_id: ID string, e.g., 'L9-1', 'L9-2'.
    Returns:
        Dict with scenario_id and parameters if found, else None.
    """
    for scenario in orthogonal_scenarios:
        if scenario['scenario_id'] == scenario_id:
            return {
                'scenario_id': scenario['scenario_id'],
                'parameters': scenario['parameters'].copy(),
                'metadata': scenario['metadata'].copy()
            }
    return None