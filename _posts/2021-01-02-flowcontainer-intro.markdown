---
layout:     post
title:      "flowcontainer: ����python3��pcap��������������Ϣ��ȡ��"
subtitle:   " flowcontainerʹ��˵��"
date:       2021-01-02
author:     "���"
header-img: "img/post-bg-os-metro.jpg"
catalog: true
tags:
	- python3
	- ������������
	- ������������
	- pcap
---

> ժҪ��Ϣ

# �����
flowcontainer�ǻ���python3����������������Ϣ��ȡ�⣬�Է���������������ķ������񡣸���pcap�ļ����ÿ����ȡpcap���е����������Ϣ����������Ϣ����������Դ�˿ڡ�ԴIP��Ŀ��IP��Ŀ�Ķ˿ڡ�IP���ݰ��ĳ������С�IP���ݼ��ĵ���ʱ�����С���Ч�غ������Լ���Ӧ��Ч�غɵĵ���ʱ�����С�����չ��Ϣ������IP���ݰ������ˣ���Щtcp/udp�غɲ�Ϊ0�����ݰ���ͳ�Ƶ���Ч�غ��������档���߼����ã���չ�Ժ͸����Ըߡ�
# ���͵�ַ
[flowcontainer: ����python3����������������Ϣ��ȡ��](https://blog.csdn.net/jmh1996/article/details/107148871)

url: https://blog.csdn.net/jmh1996/article/details/107148871

��github��ʱ����markdown����Ĺ�ʽ����������Ʋ����ͣ���ȡ���õ��ĵ��Ķ����顿
# ��İ�װ
���°棺
```bash
pip3 install git+https://github.com/jmhIcoding/flowcontainer.git
```
�ȶ��棺
```bash
pip3 install flowcontainer
```
# ��Ļ���

- python 3
- numpy>=18.1
- ϵͳ��װ��wireshark�����°汾��3.0.0��,����tshark���ڵ�Ŀ¼��ӵ�ϵͳ�Ļ���Ŀ¼����װ��wireshark�ͻ�˳����tsharkҲ��װ�á�

**���ֻ����ȡ���Ķ˿ںš��������еȻ�����Ϣ��tshark�İ汾��ֻ�����2.6.0���ɡ�**

**�����Ҫ��ȡtls��sni,��ôtshark�İ汾��Ҫ����3.0.0��**

**�����Ҫ��ȡupd.payload,��ôtshark�İ汾��Ҫ����3.3.0**

<font color="red" >
<bold>���⣬��ȷ�����нű���shell��������pycharm��vscode�����shell���ܹ���ȷ���� tshark ! �������һ������</bold> </font>

# �����ٶ�
50G���ҵ�����2��Сʱ���Ҽ��������������Ϣ����ȡ��5G���ҵ�����12���Ӽ��ɽ�����ϡ�
# ���������Լ��ų�
- ���Ҳ����ļ��Ĵ���
���������1. ���pcap��·���Ƿ���ȷ�����ʹ�þ���·��  2. ��鵱ǰshell�ܷ��tshark ��ȷ������������tshark����·����3. ���tshark�汾���Ƿ���2.6.0���ϡ�

��ISSUE ��л������
- �� ValueError: invalid literal for int() with base 10: ����
�쳣�����
```shell
if int(packet[9]) != 0:
ValueError: invalid literal for int() with base 10: ''
```
���������1. ��extract��������ʱ��ָ��filterΪ`tcp or udp` ��������Ϊpcap��������˷�tcp/udp��packet�����¶˿���Ϣ�޷�������λ��**2. tshark�������ͬһ���ֶΣ�������ȡ��Ρ����������extensions�����udp/tcp�ĳ��ȡ�ip���ȡ�ip��ַ���˿ں���������ȡ��**

��ISSUE ��л������

- �������⣺ ����github�ύissue,Ȼ���ϴ�����������ݰ��͵������̷��������⡣
# ���ʹ��
ʾ�����룺
ֱ�ӵ���extract������Ȼ�����pcap��·�����ɡ�



- ��pcap�ļ���ͬʱ���ù��˹������չ����

**flowcontainerĬ���˳��ش����������ݰ���mdns��ssdp��icmp���ݰ���Ĭ��ֻ����IP���ݰ���
flowcontainerĬ����ȡ���ģ�ԴIP��Դ�˿ڣ�Ŀ��IP��Ŀ�Ķ˿ڣ�IP�������У�IP������ʱ�����У�������ʱ�����������ʱ������غɳ������У��غɵ���ʱ�����С�**


`extract`��������3��������`infile,filter,extension`��

����:
`infile` ���ڱ�ʶpcap�ļ�·����
`filter` ������Ӱ����˹��򣬹��˹����������﷨������wireshark�ϸ񱣳�һ�¡�����Ϊ�ա�
`extension` ��������Ҫ��ȡ�Ķ������չ�ֶΣ��ֶ�������﷨����Ҳ��wireshark�ϸ񱣳�һ�¡�����Ϊ�ա�

**flowcontainer ����wireshark����������չ�ֶ���ȡ������X509֤�顢SNI��SSL��ciphersuites��tcp�غɡ�udp�غɡ�ipid�ֶεȵȡ�**

```python
__author__ = 'dk'
from flowcontainer.extractor import extract
result = extract(r"1592754322_clear.pcap",filter='',extension=["tls.handshake.extensions_server_name","tls.handshake.ciphersuite"])
```
- ��ȡpcap����������Ϣ

extract�ķ���ֵ��һ���ֵ䡣
ÿ������key������key ��һ��Ԫ�飺`��pcap�ļ����������Э�飬����ID�ţ�`�����磺`('1592754322_clear.pcap', 'tcp', '1')`

- ʹ��forѭ������������
```python
for key in result:
    ### The return vlaue result is a dict, the key is a tuple (filename,procotol,stream_id)
    ### and the value is an Flow object, user can access Flow object as flowcontainer.flows.Flow's attributes refer.

    value = result[key]
    print('Flow {0} info:'.format(key))
 ```
- **��ȡ����ԴIP**
 ```python
    ## access ip src
    print('src ip:',value.src)
 ```
- **��ȡ����Ŀ��IP**
 ```python
    ## access ip dst
    print('dst ip:',value.dst)
 ```
- **��ȡ���Ķ˿���Ϣ**
```python
    ## access srcport
    print('sport:',value.sport)
    ## access_dstport
    print('dport:',value.dport)
```

- **�����غɳ������к͵���ʱ������**
```python
    ## access payload packet lengths
    print('payload lengths :',value.payload_lengths)
    ## access payload packet timestamps sequence:
    print('payload timestamps:',value.payload_timestamps)
```
���������Ǵ������ŵģ����������ڱ�ʶ���ݰ��ǿͻ��˷�������˻����ɷ���˷����ͻ��ˡ�������ʶC->S�����ݰ���������ʶS->C �����ݰ���

- **�������Ŀ�ʼʱ��ͽ���ʱ��**

```python
	print('start timestamp :',value.time_start)
	print('end timestamp :',value.time_end)
```
��Ҫע����ǣ��������Ŀ�ʼʱ��ͽ���ʱ���ǻ���Ĭ�ϵ�ʱ��� **��Ŀǰ����Ч�غ����е�ʱ�����������IP���ݰ���ʱ�����** ������ġ�����`time_start` ͨ��`min(value.timestamps)` �õ�����`time_end` ͨ�� `max(value.timestamps)`�õ���

- **����IP���������к͵���ʱ������**

������к��غ����е��������ڣ��غ�������tcp/udp�غɲ�Ϊ�յ�tcp/udp�غ����С�IP�����л����Щ���ְ������غɵ�tcp/udp��Ҳͳ�ƽ�����

```python
    ## access ip packet lengths, (including packets with zero payload, and ip header)
    print('ip packets lengths:',value.ip_lengths)
    ## access ip packet timestamp sequence, (including packets with zero payload)
    print('ip packets timestamps:',value.ip_timestamps)
```
���������Ǵ������ŵģ����������ڱ�ʶ���ݰ��ǿͻ��˷�������˻����ɷ���˷����ͻ��ˡ�������ʶC->S�����ݰ���������ʶS->C �����ݰ���

- **����Ĭ��������Ϣ��Ĭ�����غ�������Ϣ**
```python
    ## access default lengths sequence, the default length sequences is the payload lengths sequences
    print('default length sequence:',value.lengths)
    ## access default timestamp sequence, the default timestamp sequence is the payload timestamp sequences
    print('default timestamp sequence:',value.timestamps)
```
���������Ǵ������ŵģ����������ڱ�ʶ���ݰ��ǿͻ��˷�������˻����ɷ���˷����ͻ��ˡ�������ʶC->S�����ݰ���������ʶS->C �����ݰ���

- **������չ�ֶ�**
```python
    ##access sni of the flow if any else empty str
    print('extension:',value.extension)
```
ֵ��ע����ǣ�extension��һ��dict�������key�����û��Լ�ָ����extension����ĸ���item����ÿ��key��Ӧ��value��һ��list,��ʾ�����������û���Ҫ��extension�����ֹ���ȡֵ�Լ�����չȡֵ����������IP���ݰ����ֵ��±ꡣ

��������ȡTLS���ֽ׶ε�ciphersuitesΪ�������Ǹ�extensions��������ʵ�Σ�`tls.handshake.ciphersuite` ����ô�����Ľ�������ڣ�
```python
src ip: 192.168.0.100
dst ip: 208.43.237.140
sport: 44525
dport: 443
payload lengths : [180, -1424, -1440, -190, 126, -274, 625, -1163, 31, -31]
payload timestamps: [1592993502.710372, 1592993502.710383, 1592993502.71261, 1592993502.712895, 1592993502.993892, 1592993502.993903, 1592993503.234192, 1592993504.233002, 1592993527.490709, 1592993527.49081]
ip packets lengths: [60, -60, 52, 232, -52, -1476, 52, -1492, 52, -242, 52, 178, -52, -326, 52, 677, -52, -1215, 52, -52, 52, 83, -52, 52, -83, 40, -52, 40]
ip packets timestamps: [1592993502.710358, 1592993502.710364, 1592993502.710366, 1592993502.710372, 1592993502.710377, 1592993502.710383, 1592993502.710386, 1592993502.71261, 1592993502.712891, 1592993502.712895, 1592993502.712898, 1592993502.993892, 1592993502.993895, 1592993502.993903, 1592993502.993906, 1592993503.234192, 1592993503.234202, 1592993504.233002, 1592993504.233179, 1592993518.51743, 1592993518.517824, 1592993527.490709, 1592993527.490712, 1592993527.490716, 1592993527.49081, 1592993527.490818, 1592993527.490821, 1592993527.490824]
default length sequence: [180, -1424, -1440, -190, 126, -274, 625, -1163, 31, -31]
default timestamp sequence: [1592993502.710372, 1592993502.710383, 1592993502.71261, 1592993502.712895, 1592993502.993892, 1592993502.993903, 1592993503.234192, 1592993504.233002, 1592993527.490709, 1592993527.49081]
start timestamp:1592993502.710372, end timestamp :1592993527.49081
extension: {'tls.handshake.ciphersuite': [('49195,49196,52393,49199,49200,52392,49161,49162,49171,49172,156,157,47,53', 3), ('49195', 5)]}

```
extension��һ���ֵ䣬key���Ǵ����`tls.handshake.ciphersuite`����value��һ��list��list��ÿ��Ԫ���Ǹ�tuple������tuple[0] �Ǿ����ȡֵ��tuple[1]��ʾ��ȡֵ���������ڼ���IP���ݰ����֡�tuple[1]���λ����Ϣ��������ģ�

��ΪSSL���ֽ׶ε�ciphersuites������client ��server�ṩ��ciphersuites��Ҳ�з���������ѡ���ciphersuites����Ҫͨ�������ж�ciphersuites����packet�ķ���outgoing ����incoming�� ����֪��ciphersuites��������һ���ȡֵ��

�����ʵ�����棬list������Ԫ�ء���һ����`('49195,49196,52393,49199,49200,52392,49161,49162,49171,49172,156,157,47,53', 3)`����ʾ
�����������棬��3��IP���ݰ�������ciphersuites,Ȼ��ciphersuites��ȡֵ��`'49195,49196,52393,49199,49200,52392,49161,49162,49171,49172,156,157,47,53'`��ͨ���鿴ip�������п�֪������������232���±��0������������һ���ɿͻ��˷��͸�����˵�outgoing���ݰ����������c2s�ļ����׼���
�ڶ���ȡֵ`('49195', 5)` ,��ʾ�������ĵ�5��IP���ݰ�������ciphersuits������ȡֵΪ49195����IP�������п�֪����Ӧ������ -1476������һ���ɷ�������Ӧ���������ݰ���������ciphersuits���Ƿ�����ѡ��ļ����׼���

������չ�ֶΣ�

| �ֶ��� | extensionȡֵ |��ע|
|--|--|--|
| sni | tls.handshake.extensions_server_name |tshark�汾 $\ge$ 3.0.0|
|ssl��cipher_suits|tls.handshake.ciphersuite|tshark�汾 $\ge$ 3.0.0|
|x509֤��|tls.handshake.certificate|tshark�汾 $\ge$ 3.0.0|
|udp�غ�|udp.payload|tshark�汾 $\ge$ **3.3.0**|
|tcp�غ�|tcp.payload|��|

���⣬tshark�������ͬһ���ֶΣ�������ȡ��Ρ�**���������extensions�����udp/tcp�ĳ��ȡ�ip���ȡ�ip��ַ���˿ںŵ�Ĭ����ȡ���ֶ���������ȡ���������ֱ�������Ĵ���**
# ʾ�������
���룺
```python
__author__ = 'dk'
from flowcontainer.extractor import extract
result = extract(r"1592993485_clear.pcap",filter='',extension=['tls.handshake.ciphersuite'])

for key in result:
    ### The return vlaue result is a dict, the key is a tuple (filename,procotol,stream_id)
    ### and the value is an Flow object, user can access Flow object as flowcontainer.flows.Flow's attributes refer.

    value = result[key]
    print('Flow {0} info:'.format(key))
    ## access ip src
    print('src ip:',value.src)
    ## access ip dst
    print('dst ip:',value.dst)
    ## access srcport
    print('sport:',value.sport)
    ## access_dstport
    print('dport:',value.dport)
    ## access payload packet lengths
    print('payload lengths :',value.payload_lengths)
    ## access payload packet timestamps sequence:
    print('payload timestamps:',value.payload_timestamps)
    ## access ip packet lengths, (including packets with zero payload, and ip header)
    print('ip packets lengths:',value.ip_lengths)
    ## access ip packet timestamp sequence, (including packets with zero payload)
    print('ip packets timestamps:',value.ip_timestamps)

    ## access default lengths sequence, the default length sequences is the payload lengths sequences
    print('default length sequence:',value.lengths)
    ## access default timestamp sequence, the default timestamp sequence is the payload timestamp sequences
    print('default timestamp sequence:',value.timestamps)

    print('start timestamp:{0}, end timestamp :{1}'.format(value.time_start,value.time_end))
    ##access sni of the flow if any else empty str
    print('extension:',value.extension)

print(len(result))
```

������δ������ȡ���Ļ�����Ϣ��ͬʱ��ȡssl����sni��ʾ�������
```python
Reading 1592993485_clear.pcap...
Flow ('1592993485_clear.pcap', 'tcp', '0') info:
src ip: 192.168.0.100
dst ip: 208.43.237.140
sport: 44524
dport: 443
payload lengths : [180, -1388, -1448, -216, 126, -274, 625, -1163, 361, -1092, 361, -1092, 361, -1092, 351, -888, 672, 34, -672, 935, 34, -672, 877, 34, -672]
payload timestamps: [1592993502.710375, 1592993502.718662, 1592993502.718675, 1592993502.993874, 1592993502.993886, 1592993502.993898, 1592993502.993909, 1592993503.234205, 1592993504.233183, 1592993504.233191, 1592993510.790214, 1592993510.790245, 1592993511.56349, 1592993511.563504, 1592993511.908443, 1592993511.90846, 1592993513.319068, 1592993513.31908, 1592993513.319091, 1592993517.633375, 1592993517.769362, 1592993518.085971, 1592993532.423575, 1592993532.576553, 1592993532.725287]
ip packets lengths: [60, -60, 52, 232, -52, -1440, 52, -1500, 52, -268, 52, 178, -52, -326, 52, 677, -52, -1215, 52, 413, -52, -1144, 52, 413, -52, -1144, 52, 413, -52, -1144, 52, 403, -52, -940, 52, 724, -52, 86, -52, -724, 52, 987, -52, 86, -52, -724, 52, 929, -52, 86, -52, -724, 52, -52, 52, 52, -52]
ip packets timestamps: [1592993502.710348, 1592993502.710361, 1592993502.710369, 1592993502.710375, 1592993502.71038, 1592993502.718662, 1592993502.718672, 1592993502.718675, 1592993502.718798, 1592993502.993874, 1592993502.993883, 1592993502.993886, 1592993502.993889, 1592993502.993898, 1592993502.993901, 1592993502.993909, 1592993502.994026, 1592993503.234205, 1592993503.234209, 1592993504.233183, 1592993504.233187, 1592993504.233191, 1592993504.233196, 1592993510.790214, 1592993510.790238, 1592993510.790245, 1592993510.790252, 1592993511.56349, 1592993511.563498, 1592993511.563504, 1592993511.563511, 1592993511.908443, 1592993511.908448, 1592993511.90846, 1592993512.333793, 1592993513.319068, 1592993513.319074, 1592993513.31908, 1592993513.319085, 1592993513.319091, 1592993513.319097, 1592993517.633375, 1592993517.769339, 1592993517.769362, 1592993517.769368, 1592993518.085971, 1592993518.086312, 1592993532.423575, 1592993532.576529, 1592993532.576553, 1592993532.576559, 1592993532.725287, 1592993532.725293, 1592993547.830937, 1592993547.83094, 1592993552.653862, 1592993552.899427]
default length sequence: [180, -1388, -1448, -216, 126, -274, 625, -1163, 361, -1092, 361, -1092, 361, -1092, 351, -888, 672, 34, -672, 935, 34, -672, 877, 34, -672]
default timestamp sequence: [1592993502.710375, 1592993502.718662, 1592993502.718675, 1592993502.993874, 1592993502.993886, 1592993502.993898, 1592993502.993909, 1592993503.234205, 1592993504.233183, 1592993504.233191, 1592993510.790214, 1592993510.790245, 1592993511.56349, 1592993511.563504, 1592993511.908443, 1592993511.90846, 1592993513.319068, 1592993513.31908, 1592993513.319091, 1592993517.633375, 1592993517.769362, 1592993518.085971, 1592993532.423575, 1592993532.576553, 1592993532.725287]
start timestamp:1592993502.710375, end timestamp :1592993532.725287
extension: {'tls.handshake.ciphersuite': [('49195,49196,52393,49199,49200,52392,49161,49162,49171,49172,156,157,47,53', 3), ('49195', 5)]}
Flow ('1592993485_clear.pcap', 'tcp', '1') info:
src ip: 192.168.0.100
dst ip: 208.43.237.140
sport: 44525
dport: 443
payload lengths : [180, -1424, -1440, -190, 126, -274, 625, -1163, 31, -31]
payload timestamps: [1592993502.710372, 1592993502.710383, 1592993502.71261, 1592993502.712895, 1592993502.993892, 1592993502.993903, 1592993503.234192, 1592993504.233002, 1592993527.490709, 1592993527.49081]
ip packets lengths: [60, -60, 52, 232, -52, -1476, 52, -1492, 52, -242, 52, 178, -52, -326, 52, 677, -52, -1215, 52, -52, 52, 83, -52, 52, -83, 40, -52, 40]
ip packets timestamps: [1592993502.710358, 1592993502.710364, 1592993502.710366, 1592993502.710372, 1592993502.710377, 1592993502.710383, 1592993502.710386, 1592993502.71261, 1592993502.712891, 1592993502.712895, 1592993502.712898, 1592993502.993892, 1592993502.993895, 1592993502.993903, 1592993502.993906, 1592993503.234192, 1592993503.234202, 1592993504.233002, 1592993504.233179, 1592993518.51743, 1592993518.517824, 1592993527.490709, 1592993527.490712, 1592993527.490716, 1592993527.49081, 1592993527.490818, 1592993527.490821, 1592993527.490824]
default length sequence: [180, -1424, -1440, -190, 126, -274, 625, -1163, 31, -31]
default timestamp sequence: [1592993502.710372, 1592993502.710383, 1592993502.71261, 1592993502.712895, 1592993502.993892, 1592993502.993903, 1592993503.234192, 1592993504.233002, 1592993527.490709, 1592993527.49081]
start timestamp:1592993502.710372, end timestamp :1592993527.49081
extension: {'tls.handshake.ciphersuite': [('49195,49196,52393,49199,49200,52392,49161,49162,49171,49172,156,157,47,53', 3), ('49195', 5)]}
```

# ��װ����ͳ��
��pypi���Բ�ѯ��ÿ����ͨ��pip��װflowcontainer��������Ϣ��
|������| �·� |
|--|--|
|  1944|2020-11  |
|1315|2020-10|
|1196|2020-09|


