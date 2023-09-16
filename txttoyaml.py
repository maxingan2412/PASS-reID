import yaml

def convert_requirements_to_yaml(input_file, output_file):
    # 读取 requirements.txt 文件内容
    with open(input_file, 'r') as file:
        requirements_lines = file.readlines()

    # 将 requirements.txt 内容转换成字典格式
    requirements_dict = {'dependencies': []}
    for line in requirements_lines:
        line = line.strip()
        if line:
            requirements_dict['dependencies'].append(line)

    # 将字典写入 requirements.yml 文件
    with open(output_file, 'w') as file:
        yaml.dump(requirements_dict, file)

if __name__ == "__main__":
    input_file = "requirements.txt"
    output_file = "requirements.yml"
    convert_requirements_to_yaml(input_file, output_file)
