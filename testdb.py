import mysql.connector

# try:
#     conn = mysql.connector.connect(
#         host="13.212.22.175",   # Địa chỉ host của database đã deploy
#         user="house_sale_admin",        # Tên đăng nhập
#         password="HouseSale2024!",    # Mật khẩu
#         database="house_sale"       # Tên cơ sở dữ liệu
#     )
#     if conn.is_connected():
#         print("Successfully connected to the database")
#     conn.close()
# except mysql.connector.Error as err:
#     print(f"Error: {err}")
    
def query_database(bedrooms=None, bedrooms_condition=None, floors=None, floors_condition=None, toilets=None, toilets_condition=None, area=None, area_condition=None):
    # Cập nhật thông tin kết nối
    conn = mysql.connector.connect(
        host="13.212.22.175",   # Địa chỉ host của database đã deploy
        user="house_sale_admin",        # Tên đăng nhập
        password="HouseSale2024!",    # Mật khẩu
        database="house_sale"       # Tên cơ sở dữ liệu
    )
    cursor = conn.cursor()
    
    # Chỉnh sửa query để sử dụng các tham số
    base_query = "SELECT propertyId FROM Properties WHERE 1=1 "
    
    params = []
    
    if bedrooms is not None and bedrooms_condition is not None:
        if bedrooms_condition == "less_than":
            base_query += " AND bedrooms <= %s"
        elif bedrooms_condition == "more_than":
            base_query += " AND bedrooms >= %s"
        elif bedrooms_condition == "equal":
            base_query += " AND bedrooms = %s"
        params.append(bedrooms)
    
    if floors is not None and floors_condition is not None:
        if floors_condition == "less_than":
            base_query += " AND floors <= %s"
        elif floors_condition == "more_than":
            base_query += " AND floors >= %s"
        elif floors_condition == "equal":
            base_query += " AND floors = %s"
        params.append(floors)

    if toilets is not None and toilets_condition is not None:
        if toilets_condition == "less_than":
            base_query += " AND toilets <= %s"
        elif toilets_condition == "more_than":
            base_query += " AND toilets >= %s"
        elif toilets_condition == "equal":
            base_query += " AND toilets = %s"
        params.append(toilets)

    if area is not None and area_condition is not None:
        if area_condition == "less_than":
            base_query += " AND area <= %s"
        elif area_condition == "more_than":
            base_query += " AND area >= %s"
        elif area_condition == "equal":
            base_query += " AND area = %s"
        params.append(area)

    cursor.execute(base_query, tuple(params))
    results = cursor.fetchall()
    conn.close()
    return results

# Gọi hàm và hiển thị kết quả ra terminal
if __name__ == "__main__":
    floors = 2
    bathrooms = 2
    results = query_database(floors, bathrooms)
    
    for row in results:
        print(row)