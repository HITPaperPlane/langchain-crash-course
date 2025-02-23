这是面试者的一些资料
<html><body><table><tr><td>姓名</td><td>性别</td><td>高考成绩</td><td>大学</td><td>颜值</td></tr><tr><td>张老三</td><td>男</td><td>699</td><td>北京大学</td><td>帅</td></tr><tr><td>李老四</td><td>男</td><td>初中毕业，没高考，没上大学</td><td></td><td>一般般</td></tr><tr><td rowspan="2">李立</td><td>男</td><td>566</td><td>石河子大学</td><td>丑</td></tr><tr><td>女</td><td>422</td><td>郑州铁路大学</td><td>大美女</td></tr></table></body></html>
输入：
<html><body><table><tr><th colspan="2">水果</th><th>产地</th></tr><tr><td>苹果</td><td>红色</td><td>中国</td></tr><tr><td>香蕉</td><td>黄色</td><td>菲律宾</td></tr></table></body></html>
输出：
<html><body><table><tr><td>水果</td><td>水果</td><td>产地</td></tr><tr><td>苹果</td><td>红色</td><td>中国</td></tr><tr><td>香蕉</td><td>黄色</td><td>菲律宾</td></tr></table></body></html>

输入：
<html><body><table><tr><th rowspan="2" colspan="2">综合信息</th><th colspan="4">详细数据</th></tr><tr><th>项目1</th><th>项目2</th><th>项目3</th><th>项目4</th></tr><tr><td>A</td><td>B</td><td>C</td><td>D</td><td>E</td><td>F</td></tr><tr><td>G</td><td>H</td><td>I</td><td>J</td><td>K</td><td>L</td></tr></table></body></html>
输出：
<html><body><table><tr><td>综合信息</td><td>综合信息</td><td>详细数据</td><td>详细数据</td><td>详细数据</td><td>详细数据</td></tr><tr><td>综合信息</td><td>综合信息</td><td>项目1</td><td>项目2</td><td>项目3</td><td>项目4</td></tr><tr><td>A</td><td>B</td><td>C</td><td>D</td><td>E</td><td>F</td></tr><tr><td>G</td><td>H</td><td>I</td><td>J</td><td>K</td><td>L</td></tr></table></body></html>