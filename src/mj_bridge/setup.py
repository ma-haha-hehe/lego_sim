from glob import glob
from setuptools import find_packages, setup

package_name = 'mj_bridge'

setup(
    name=package_name,
    version='0.2.0',
    packages=find_packages(exclude=['test']),
    package_data={
        'mj_bridge': ['*.yaml', '*.json', '*.xml', 'assets/*'],
    },
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'BENCHMARK.md']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/examples', ['../../examples/products/traffic_light.yaml']),
        ('share/' + package_name + '/mj_bridge', [
            'mj_bridge/benchmark.yaml',
            'mj_bridge/part_registry.yaml',
            'mj_bridge/product_schema.json',
            'mj_bridge/initial.yaml',
            'mj_bridge/initial_positions.yaml',
            'mj_bridge/scene_template.xml',
            'mj_bridge/panda.xml',
            'mj_bridge/hand.xml',
        ]),
    ],
    install_requires=['setuptools', 'PyYAML', 'numpy', 'mujoco>=3.2,<4'],
    zip_safe=True,
    maintainer='Wenbo Ma',
    maintainer_email='wenbo.ma@rwth-aachen.de',
    description='MuJoCo and ROS 2 bridge for repeatable LEGO manipulation experiments',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'mj_bridge = mj_bridge.mj_bridge3:main',
            'lego-bench = mj_bridge.benchmark_cli:main',
        ],
    },
)
