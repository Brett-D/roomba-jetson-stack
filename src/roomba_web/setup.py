from glob import glob

from setuptools import setup

package_name = 'roomba_web'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/www', glob('www/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='brett',
    maintainer_email='brett@example.com',
    description='Web app: live detections with distance, top-down map, exclusion zones.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'web_server = roomba_web.server:main',
            'set_password = roomba_web.set_password:main',
        ],
    },
)
