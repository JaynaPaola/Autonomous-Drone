import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'drone'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dany',
    maintainer_email='mega2207@outlook.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'odometry = drone.odometry:main',
            'controller = drone.controller:main',
            'plotter = drone.plotter:main',
            'realtime_plot = drone.realtime_plot:main'
        ],
    },
)
