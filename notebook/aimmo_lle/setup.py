from setuptools import setup, find_packages

# 의존성을 requirements.txt 파일에서 로드합니다.
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# 프로젝트에 대한 정보를 입력하세요.
setup(
    name='va',
    version='0.1.0',
    description='Low Light Enhancement',
    author='Clid-Jeon',
    author_email='clid@aimmo.co.kr',
    license='Your License',
    install_requires=requirements,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Your License Here',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    packages=find_packages(exclude=('demo', 'docker')),
    include_package_data=True,
)