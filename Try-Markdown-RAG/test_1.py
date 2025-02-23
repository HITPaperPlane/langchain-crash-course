from bs4 import BeautifulSoup

def unmerge_table_cells(html_table_string):
    """
    处理HTML表格字符串，拆分合并单元格，返回无合并单元格的HTML表格字符串。

    Args:
        html_table_string: 包含合并单元格的HTML表格字符串。

    Returns:
        没有合并单元格的HTML表格字符串。
    """
    soup = BeautifulSoup(html_table_string, 'html.parser')
    table = soup.find('table')
    if not table:
        return html_table_string  # 如果没有找到表格，则原样返回

    unmerged_rows = []
    carry_over_cells = []

    table_body = table.find('tbody') # 尝试查找 tbody
    if table_body:
        rows = table_body.find_all('tr') # 如果有 tbody，则在 tbody 中查找 tr
    else:
        rows = table.find_all('tr')      # 如果没有 tbody，则直接在 table 中查找 tr


    for row in rows: # 使用 rows 变量
        current_unmerged_row = []
        current_col_index = 0

        # 处理 carry-over cells
        next_carry_over_cells = []
        for content, remaining_rows, colspan in carry_over_cells:
            for _ in range(colspan):
                current_unmerged_row.append(content)
            current_col_index += colspan
            if remaining_rows > 1:
                next_carry_over_cells.append((content, remaining_rows - 1, colspan))
        carry_over_cells = next_carry_over_cells

        # 处理当前行的 cells
        cells = row.find_all('td')
        if not cells: # 处理th的情况, 更加通用
            cells = row.find_all('th')

        for cell in cells:
            content = cell.get_text(separator='').strip() # 使用separator=''来避免BeautifulSoup默认添加换行符
            colspan = int(cell.get('colspan', 1))
            rowspan = int(cell.get('rowspan', 1))

            for _ in range(colspan):
                current_unmerged_row.append(content)
            current_col_index += colspan

            if rowspan > 1:
                carry_over_cells.append((content, rowspan - 1, colspan))

        unmerged_rows.append(current_unmerged_row)

    # 构建新的HTML表格
    new_table_soup = BeautifulSoup("<html><body><table></table></body></html>", 'html.parser')
    new_table = new_table_soup.find('table')

    for unmerged_row_data in unmerged_rows:
        new_tr = new_table_soup.new_tag('tr')
        for cell_content in unmerged_row_data:
            new_td = new_table_soup.new_tag('td')
            new_td.string = cell_content
            new_tr.append(new_td)
        new_table.append(new_tr)

    return str(new_table_soup) # 外层用html标签包裹


# 示例markdown表格HTML字符串
markdown_table_html = """<html><body><table><tr><td>姓名</td><td>性别</td><td>高考成绩</td><td>大学</td><td>颜值</td></tr><tr><td>张老三</td><td>男</td><td>699</td><td>北京大学</td><td>帅</td></tr><tr><td>李老四</td><td>男</td><td>初中毕业，没高考，没上大学</td><td></td><td>一般般</td></tr><tr><td rowspan="2">李立</td><td>男</td><td>566</td><td>石河子大学</td><td>丑</td></tr><tr><td>女</td><td>422</td><td>郑州铁路大学</td><td>大美女</td></tr></table></body></html>"""
m2 = """<html><body><table><tr><th colspan="2">水果</th><th>产地</th></tr><tr><td>苹果</td><td>红色</td><td>中国</td></tr><tr><td>香蕉</td><td>黄色</td><td>菲律宾</td></tr></table></body></html>"""
m3 = """<html><body><table><tr><th rowspan="2" colspan="2">综合信息</th><th colspan="4">详细数据</th></tr><tr><th>项目1</th><th>项目2</th><th>项目3</th><th>项目4</th></tr><tr><td>A</td><td>B</td><td>C</td><td>D</td><td>E</td><td>F</td></tr><tr><td>G</td><td>H</td><td>I</td><td>J</td><td>K</td><td>L</td></tr></table></body></html>"""
# 处理并输出结果
unmerged_html = unmerge_table_cells(m3)
print(unmerged_html)