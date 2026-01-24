import xml.etree.ElementTree as ET

def parse_calib_file(file_path):
    """
    输入标定文件路径，解析标定数据
    用法示例：
        file_path = "path/to/calib.mfa"
    """
    tree = ET.parse(file_path)
    root = tree.getroot()

    
    rot_mat = root.find('RTmatRgb2robot/RotMat')
    t_vec = root.find('RTmatRgb2robot/TVec')
    result = []
    if rot_mat is not None:
        for key in sorted(rot_mat.attrib):
            if key.startswith('r'):
                result.append(f"{key}={rot_mat.attrib[key]}")
    if t_vec is not None:
        for key in sorted(t_vec.attrib):
            if key.startswith('t'):
                result.append(f"{key}={t_vec.attrib[key]}")
    return '\n'.join(result)

def parse_from_string(xml_string):
    """
    从标定文件字符串中提取数据
    用法示例：
        将标定文件RTmatRgb2robot部分复制在字符串中
        xml_string = /"/"/"<RTmatRgb2robot>
        <RotMat r00="0.71730157845489351" r01="-0.69666075271742278" r02="-0.011926490234436834" 
        r10="-0.69672998243450446" r11="-0.71732888876268619" r12="-0.0025684476449435497" 
        r20="-0.0067658793170659376" r21="0.010151894881441227" r22="-0.99992557818438821" help="rgb相机到机器人旋转矩阵" />
        <TVec t0="607.54687187762647" t1="-857.99864513969658" t2="805.87263333128044" help="rgb相机到机器人平移向量" />
        </RTmatRgb2robot>/"/"/"
    """
    root = ET.fromstring(xml_string)
    rot_mat = root.find('RotMat')
    t_vec = root.find('TVec')
    result = []
    if rot_mat is not None:
        for key in sorted(rot_mat.attrib):
            if key.startswith('r'):
                result.append(f"{key}={rot_mat.attrib[key]}")
    if t_vec is not None:
        for key in sorted(t_vec.attrib):
            if key.startswith('t'):
                result.append(f"{key}={t_vec.attrib[key]}")
    return '\n'.join(result)

#-------------------------------#-------------------------------#
def file():
    file_path = "calib.mfa"
    result = parse_calib_file(file_path)
    print(result)
def string():
    xml_string = """<RTmatRgb2robot>
          <RotMat r00="0.70954548185103394" r01="-0.70458504844781489" r02="0.010252740539697287" r10="-0.70440624321192413" r11="-0.70960540970529506" r12="-0.016492635964088662" r20="0.018895864861063563" r21="0.0044801808859382215" r22="-0.99981141935386086" help="rgb相机到机器人旋转矩阵" />
          <TVec t0="706.79987005314251" t1="-890.77854779051279" t2="910.95843390205141" help="rgb相机到机器人平移向量" />
        </RTmatRgb2robot>"""
    # xml_string = """<RTmatRgb2robot> date: 0605 上午
    #       <RotMat r00="0.67542289786461751" r01="-0.73742679123009436" r02="-0.002374155902434903" r10="-0.73742589722983221" r11="-0.67542659199983279" r12="0.0014017539969910192" r20="-0.0026372589821529405" r21="0.00080398729977557615" r22="-0.99999619922751848" help="rgb相机到机器人旋转矩阵" />
    #       <TVec t0="-1085.0079701665638" t1="-40.019275650096574" t2="829.6882943243811" help="rgb相机到机器人平移向量" />
    #     </RTmatRgb2robot>"""
    result = parse_from_string(xml_string)
    print(result)

if __name__ == "__main__":
    # file()
    string()
