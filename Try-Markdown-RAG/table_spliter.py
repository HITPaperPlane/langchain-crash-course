from bs4 import BeautifulSoup

def parse_html_table(html):
    """解析HTML表格，返回表头和数据行"""
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find('table')
    if not table:
        return [], []
    rows = []
    for tr in table.find_all('tr'):
        cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
        if cells:
            rows.append(cells)
    return (rows[0], rows[1:]) if rows else ([], [])

def table_to_md(headers, rows):
    """将表格转换为Markdown格式字符串（用于长度计算）"""
    if not headers and not rows:
        return ""
    header_row = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = [
        "| " + " | ".join(row[:len(headers)]) + " |"
        for row in rows
    ]
    return "\n".join([header_row, separator] + body)

def split_row_by_columns(headers, row, chunk_size):
    """按列拆分单行数据，返回（子表列表，超大单元格列表）"""
    split_tables = []
    oversized_cells = []
    
    if not headers or not row:
        return split_tables, oversized_cells

    # 提取行标题信息
    row_header_title = headers[0] if headers else ""
    row_header_value = row[0] if row else ""
    
    # 检查行标题列本身是否可容纳
    base_headers = [row_header_title]
    base_row = [row_header_value]
    base_md = table_to_md(base_headers, [base_row])
    if len(base_md) > chunk_size:
        oversized_cells.append(f"{row_header_title}-{row_header_value}")
        split_tables.append((base_headers, [""]))  # 用空值代替
        return split_tables, oversized_cells
    
    # 处理其他列（从索引1开始）
    current_col = 1
    while current_col < len(headers):
        max_cols = 0
        found_valid = False
        
        # 寻找能容纳的最大列数（包含行标题）
        for end in range(current_col + 1, len(headers) + 1):
            current_headers = [row_header_title] + headers[current_col:end]
            current_row = [row_header_value] + (row[current_col:end] if len(row) > current_col else [row_header_value] + [''] * (end - current_col))
            current_row += [''] * (len(current_headers) - len(current_row))
            md = table_to_md(current_headers, [current_row])
            
            if len(md) <= chunk_size:
                max_cols = end - current_col
                found_valid = True
            else:
                break
        
        if found_valid:
            # 生成有效子表
            sub_headers = [row_header_title] + headers[current_col:current_col + max_cols]
            sub_row = [row_header_value] + (row[current_col:current_col + max_cols] if len(row) > current_col else [row_header_value] + [''] * max_cols)
            sub_row += [''] * (len(sub_headers) - len(sub_row))
            split_tables.append((sub_headers, sub_row))
            current_col += max_cols
        else:
            # 处理超大单元格
            col_header = headers[current_col] if current_col < len(headers) else ""
            cell_value = row[current_col] if current_col < len(row) else ""
            
            # 尝试单独添加该列
            temp_headers = [row_header_title, col_header]
            temp_row = [row_header_value, cell_value]
            md = table_to_md(temp_headers, [temp_row])
            
            if len(md) <= chunk_size:
                split_tables.append((temp_headers, temp_row))
            else:
                # 记录超大单元格
                oversized_cells.append(f"{row_header_title}-{row_header_value}-{col_header}-{cell_value}")
                split_tables.append((temp_headers, [row_header_value, ""]))
            
            current_col += 1
    
    return split_tables, oversized_cells

def split_table(html, chunk_size):
    """主拆分函数，返回（结构化列表，超大单元格列表）"""
    headers, data_rows = parse_html_table(html)
    oversized_cells = []
    
    if not headers and not data_rows:
        return [], []
    
    # 检查完整表格是否可容纳
    full_md = table_to_md(headers, data_rows)
    if len(full_md) <= chunk_size:
        return [[headers] + data_rows], []
    
    result = []
    current_chunk = []
    
    def add_chunk(h, rs):
        """添加一个子表到结果"""
        if rs:
            result.append([h] + rs)
    
    for row in data_rows:
        # 尝试将当前行加入临时块
        temp_md = table_to_md(headers, current_chunk + [row])
        
        if len(temp_md) <= chunk_size:
            current_chunk.append(row)
        else:
            if current_chunk:
                add_chunk(headers, current_chunk)
                current_chunk = []
            
            # 检查单行是否可容纳
            single_row_md = table_to_md(headers, [row])
            if len(single_row_md) <= chunk_size:
                current_chunk = [row]
            else:
                # 按列拆分
                sub_tables, cells = split_row_by_columns(headers, row, chunk_size)
                oversized_cells.extend(cells)
                for sub_table in sub_tables:
                    add_chunk(sub_table[0], [sub_table[1]])
    
    # 处理最后剩余的缓存
    if current_chunk:
        add_chunk(headers, current_chunk)
    
    return result, oversized_cells

# 测试实例
if __name__ == "__main__":
    # 测试案例：需要按列拆分并包含行标题
    test_html = """
    <html><body><table>
        <tr><td>序号</td><td>环境项目</td><td>标准要求</td></tr>
        <tr><td>1</td><td>照明条件</td><td>作业现场照明不低于150lx</td></tr>
        <tr><td>2</td><td>通风条件</td><td>确保空气流通，有害气体浓度符合标准</td></tr>
    </table></body></html>
    """
    tables, big_cells = split_table(test_html, 190)
    print("测试案例输出:")
    print("拆分结果:")
    for table in tables:
        print(table)
    print("超大单元格:", big_cells)

    # 测试超大单元格场景
    html4 = """
    <html><body><table>
        <tr><th>项目</th><th>内容</th></tr>
        <tr><td>地址</td><td>ThisIsAnExtremelyLongAddressThatExceedsTheChunkSizeWhenConvertedToMarkdownFormat</td></tr>
    </table></body></html>
    """
    tables, big_cells = split_table(html4, 80)
    print("\n超大单元格测试:")
    print("拆分结果:")
    for table in tables:
        print(table)
    print("超大单元格:", big_cells)
    print(len("汉字"))