import re
import string
import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
from wordcloud import WordCloud
import squarify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
from underthesea import word_tokenize
import joblib
# ---------------------------
uploaded_file_rs = pd.read_csv("data/df_res.csv")
df_rev1 = pd.read_csv("data/df_rev1.csv")
df_rev2 = pd.read_csv("data/df_rev2.csv")
df_rev = pd.concat([df_rev1, df_rev2])

file = open('data/files/vietnamese-stopwords.txt', 'r', encoding="utf8")
stopwords_lst = file.read().split('\n')
file.close()

df_PNR = pd.read_csv('DF_PNR.csv')

joblib_file = "logistic_regression_model.pkl"
# Load the model from the file
loaded_model = joblib.load(joblib_file)
# Convert the text data to numerical vectors
vectorizer = TfidfVectorizer(analyzer='word', min_df=0, stop_words=stopwords_lst)
vectorizer.fit_transform(df_PNR['Comment'].tolist())

level_extreme_words = [
    "rất", "khủng khiếp", "quá", "siêu", "cực kỳ", "rất nhiều", "vô cùng", "tuyệt đối", "hoàn toàn",
    "hết sức", "thật sự", "thực sự", "chắc chắn", "rõ ràng", "đáng kể", "lớn", "nhiều"
]

level_minor_words = [
    "hơi", "chút", "một chút", "một tí", "phần nào", "hầu như", "gần như", "suýt nữa", "chỉ",
    "chỉ có", "chỉ là", "chỉ cần", "chỉ mới", "chỉ còn", "chỉ còn lại", "chỉ còn một",
    "chỉ còn vài", "chỉ còn ít", "chỉ còn một chút", "chỉ còn một tí", "chỉ còn một chút", "chỉ còn một tí",
    "chỉ còn một phần", "chỉ còn một ít", "chỉ còn một phần"
]

negative_words = [
    "kém", "tệ", "đau", "xấu", "dở", "ức","tanh","nhỏ", "ít",
    "buồn", "rối", "thô", "lâu", "chán",
    "tối", "chán", "ít", "mờ", "mỏng",
    "lỏng lẻo", "khó", "cùi", "yếu",
    "kém chất lượng", "không thích", "không thú vị", "không ổn",
    "không hợp", "không đáng tin cậy", "không chuyên nghiệp",
    "không phản hồi", "không an toàn", "không phù hợp", "không thân thiện", "không linh hoạt", "không đáng giá",
    "không ấn tượng", "không tốt", "chậm", "khó khăn", "phức tạp",
    "khó hiểu", "khó chịu", "gây khó dễ", "rườm rà", "khó truy cập",
    "thất bại", "tồi tệ", "khó xử", "không thể chấp nhận", "tồi tệ","không rõ ràng",
    "không chắc chắn", "rối rắm", "không tiện lợi", "không đáng tiền", "chưa đẹp", "không đẹp", "khiếu nại", "kiện", "bồi thường", "độc"
    , "ngộ độc", "đau bụng", "bất lịch sự", "thiếu", "thiếu trách nhiệm", "thiếu hiệu quả", "không đáp ứng nhu cầu",
    "không thân thiện", "không chu đáo", "không hỗ trợ", "không linh hoạt", "không minh bạch", "không công bằng",
    "không tôn trọng", "không có năng lực", "không có kinh nghiệm", "không có kiến thức", "không có kỹ năng",
    "không có tinh thần", "không có đạo đức", "không có phẩm chất", "không có khả năng giao tiếp",
    "không có khả năng giải quyết vấn đề", "không có khả năng làm việc nhóm", "không có khả năng thích nghi",
    "không có khả năng học hỏi", "không có khả năng sáng tạo",
    # về cảm xúc tiêu cực:
    "bực mình", "giận dữ", "ghét", "căm thù", "sợ hãi", "lo lắng", "bất an", "tuyệt vọng", "cô đơn",
    "thất vọng", "đau khổ", "bế tắc", "mệt mỏi", "chán nản", "bất lực", "vô vọng", "ngán ngẩm", "bất hạnh",
    # về tính cách tiêu cực:
    "tham lam", "ích kỷ", "gian ác", "xảo quyệt", "dối trá", "bất lương", "hèn nhát", "ngu ngốc", "bất tài",
    "bất lịch sự", "vô lễ", "kiêu ngạo", "hợm hĩnh", "bất công", "bất chính", "bất nghĩa", "bất nhân",
    # về sự kiện, tình huống tiêu cực:
    "thảm họa", "tai nạn", "bệnh tật", "chiến tranh", "khủng bố", "bạo lực", "bất công", "bất bình đẳng",
    "phân biệt đối xử", "ô nhiễm", "thiên tai", "nạn đói", "nạn dịch", "suy thoái", "khủng hoảng", "thất bại",
    "tan vỡ", "mất mát", "đau thương", "buồn bã",
    # về ngoại hình tiêu cực:
    "xấu xí", "xù xì", "nhăn nheo", "gầy gò", "béo phì", "bẩn thỉu", "hôi hám", "bất cân đối", "lệch lạc",
    # về âm thanh tiêu cực:
    "ồn ào", "ầm ĩ", "khóc than", "rên rỉ", "kêu la", "gào thét", "sột soạt", "lạo xạo", "lách cách", "leng keng",
    # về mùi vị tiêu cực:
    "hôi", "thối", "chua", "đắng", "cay", "nồng", "hăng", "ngấy", "nhạt nhẽo", "khó chịu",
    # về xúc giác tiêu cực:
    "nhám", "gồ ghề", "lạnh lẽo", "nóng bức", "ẩm ướt", "dính nhớp", "bẩn thỉu", "khó chịu", "đau đớn",
    # về sự vật tiêu cực:
    "rác rưởi", "bụi bẩn", "ô nhiễm", "bệnh tật", "thú dữ", "quái vật", "ma quỷ", "ác quỷ", "quỷ dữ",
    # về hành động tiêu cực:
    "lừa đảo", "bắt nạt", "bạo lực", "đánh đập", "tra tấn", "giết người", "cướp bóc", "phá hoại", "phỉ báng", "vu khống",
    # về trạng thái tiêu cực:
    "bất hạnh", "bất an", "bất ổn", "bất lực", "bất hạnh", "bất công", "bất bình đẳng", "bất hòa", "bất đồng",
    # về tính chất tiêu cực:
    "sai trái", "phi lý", "vô lý", "bất hợp pháp", "bất hợp lý", "bất thường", "bất thường", "bất ổn định", "bất khả thi",
    # về kết quả tiêu cực:
    "thất bại", "thất vọng", "thất lạc", "thất thoát", "thất bại", "thất bại", "thất bại", "thất bại", "thất bại",
    # về phản ứng tiêu cực:
    "tức giận", "phẫn nộ", "ghê tởm", "kinh hoàng", "sợ hãi", "lo lắng", "bất an", "tuyệt vọng", "cô đơn",
    "thất vọng", "đau khổ", "bế tắc", "mệt mỏi", "chán nản", "bất lực", "vô vọng", "ngán ngẩm", "bất hạnh",
    # Từ ngữ tiêu cực
    "thiếu chuyên nghiệp", "bất lịch sự", "thiếu trách nhiệm", "thiếu hiệu quả", "không đáp ứng nhu cầu",
    "không thân thiện", "không chu đáo", "không hỗ trợ", "không linh hoạt", "không minh bạch", "không công bằng",
    "không tôn trọng", "không an toàn", "không an toàn thực phẩm", "không an toàn môi trường"
    # Từ ngữ tiêu cực cho cửa hàng đồ ăn
    "phục vụ chậm", "giao hàng chậm", "môi trường bẩn", "bàn ghế bẩn", "nhà vệ sinh bẩn", "món ăn không ngon",
    "hương vị không ngon", "chất lượng món ăn kém", "món ăn hỏng", "đồ uống hỏng", "nguyên liệu hỏng",
    "phục vụ thiếu", "món ăn thiếu", "đồ uống thiếu", "không gian không thoải mái", "chỗ ngồi không thoải mái",
    "bầu không khí không thoải mái", "vị trí không thuận lợi", "khó tìm", "khó đặt hàng", "giá cả quá cao",
    "giá cả không hợp lý", "lỗi phục vụ", "lỗi món ăn", "lỗi đồ uống", "thái độ phục vụ không tốt", "thái độ thiếu chuyên nghiệp",
    "thái độ không thân thiện", "phục vụ không chuyên nghiệp", "nhân viên không chuyên nghiệp", "phục vụ không chu đáo",
    "nhân viên không chu đáo", "bất cẩn trong phục vụ", "bất cẩn trong chế biến", "thiếu vệ sinh an toàn thực phẩm",
    "thiếu vệ sinh môi trường", "không an toàn thực phẩm", "không an toàn môi trường", "menu không đa dạng",
    "lựa chọn không đa dạng", "món ăn không sáng tạo", "cách phục vụ không sáng tạo", "chất lượng dịch vụ kém",
    "chất lượng sản phẩm kém", "không uy tín cửa hàng", "không uy tín thương hiệu", "không an toàn thực phẩm",
    "không an toàn môi trường"
    # So sánh giá
    "đắt", "cao", "thấp", "kém", "bèo", "chát", "quá đắt", "quá rẻ",
    "đắt đỏ", "rẻ bèo", "không hợp lý", "không phải chăng", "kém giá trị", "kém chất lượng", "không đáng tiền", "lừa đảo",
    "ép giá", "bắt chẹt", "chặt chém", "không minh bạch", "không công bằng", "không rõ ràng", "lừa đảo", "ép giá", "bắt chẹt", "chặt chém"

]

positive_words = [
    "thích", "tốt", "xuất sắc", "tuyệt vời", "tuyệt hảo", "đẹp", "ổn", "ngon", "ngon miệng",
    "hài lòng", "ưng ý", "hoàn hảo", "chất lượng", "thú vị", "nhanh",
    "tiện lợi", "dễ sử dụng", "hiệu quả", "ấn tượng",
    "nổi bật", "tận hưởng", "tốn ít thời gian", "thân thiện", "hấp dẫn",
    "gợi cảm", "tươi mới", "lạ mắt", "cao cấp", "độc đáo",
    "hợp khẩu vị", "rất tốt", "rất thích", "tận tâm", "đáng tin cậy", "đẳng cấp",
    "hấp dẫn", "an tâm", "không thể cưỡng lại", "thỏa mãn", "thúc đẩy",
    "cảm động", "phục vụ tốt", "làm hài lòng", "gây ấn tượng", "nổi trội",
    "sáng tạo", "quý báu", "phù hợp", "tận tâm",
    "hiếm có", "cải thiện", "hoà nhã", "chăm chỉ", "cẩn thận",
    "vui vẻ", "sáng sủa", "hào hứng", "đam mê", "vừa vặn", "đáng tiền", "uy tín", "giá cả hợp lý", "giá cả phải chăng", "giá cả phù hợp"
    , "giá hợp lý", "giá phải chăng", "giá phù hợp", "chuyên nghiệp", "thân thiện", "chu đáo", "hỗ trợ", "linh hoạt", "đổi mới", "đúng hẹn", "đúng giờ"
    "minh bạch", "công bằng", "tôn trọng", "nhanh", "chính xác",
    # về cảm xúc tích cực:
    "vui", "hạnh phúc", "yêu thương", "biết ơn", "thỏa mãn", "tự hào", "kiêu hãnh", "an nhiên", "bình yên", "thoải mái",
    "sung sướng", "hân hoan", "rạng rỡ", "tươi cười", " phấn khởi", "thích thú", "ngạc nhiên", "ngưỡng mộ", "kính trọng",
    "trân trọng", "hy vọng", "tin tưởng", "yên tâm", "an toàn", "tự do", "dũng cảm", "kiên cường", "nhân ái", "từ bi",
    # về tính cách tích cực:
    "tốt bụng", "hiền lành", "chính trực", "trung thực", "thông minh", "tài năng", "nhân ái", "từ bi", "kiên nhẫn",
    "bao dung", "tha thứ", "độ lượng", "khiêm tốn", "lòng tốt", "nhân hậu", "vị tha", "dũng cảm", "kiên cường",
    "quyết đoán", "sáng suốt", "nhạy bén", "tư duy logic", "sáng tạo", "bền bỉ", "chuyên nghiệp", "có trách nhiệm",
    "tận tâm", "chăm chỉ", "cẩn thận", "có tổ chức", "có kỷ luật", "có tinh thần đồng đội", "hòa đồng", "thân thiện",
    "lạc quan", "tích cực", "hài hước", "thú vị", "thu hút", "quyến rũ", "lôi cuốn",
    # về sự kiện, tình huống tích cực:
    "thành công", "chiến thắng", "hạnh phúc", "lễ hội", "sinh nhật", "kỷ niệm", "tình yêu", "gia đình", "bạn bè",
    "hòa bình", "an ninh", "phát triển", "tiến bộ", "cải thiện", "phồn vinh", "thịnh vượng", "hạnh phúc", "an khang",
    "thịnh đạt", "tự do", "công bằng", "bình đẳng", "tôn trọng", "yêu thương", "chăm sóc", "giúp đỡ", "chia sẻ",
    "bao dung", "tha thứ", "độ lượng", "nhân ái", "từ bi", "lòng tốt", "nhân hậu", "vị tha", "dũng cảm", "kiên cường",
    "quyết đoán", "sáng suốt", "nhạy bén", "tư duy logic", "sáng tạo", "bền bỉ", "chuyên nghiệp", "có trách nhiệm",
    "tận tâm", "chăm chỉ", "cẩn thận", "có tổ chức", "có kỷ luật", "có tinh thần đồng đội", "hòa đồng", "thân thiện",
    "lạc quan", "tích cực", "hài hước", "thú vị", "thu hút", "quyến rũ", "lôi cuốn",
    # về ngoại hình tích cực:
    "đẹp", "xinh", "dễ thương", "quyến rũ", "lôi cuốn", "thu hút", "thanh lịch", "sang trọng", "rạng rỡ", "tươi trẻ",
    "khỏe mạnh", "tràn đầy sức sống", "bức xạ", "tỏa sáng", "quyến rũ", "năng động", "hấp dẫn", "gợi cảm", "tươi tắn",
    "rạng ngời", "tỏa nắng", "rực rỡ", "nổi bật", "ấn tượng", "lịch lãm", "phong độ", "dáng vẻ", "phong thái", "thái độ",
    # về âm thanh tích cực:
    "êm ái", "du dương", "nhạc du dương", "tiếng chim hót", "tiếng cười", "tiếng suối", "tiếng gió", "tiếng mưa",
    "tiếng đàn", "tiếng hát", "tiếng chuông", "tiếng cười", "tiếng nói chuyện vui vẻ", "tiếng động vui tai",
    # về mùi vị tích cực:
    "thơm", "ngon", "ngọt", "chua ngọt", "mùi thơm", "mùi hoa", "mùi trái cây", "mùi bánh", "mùi cà phê",
    "mùi nước hoa", "mùi sữa", "mùi đất", "mùi biển", "mùi thảo mộc", "mùi gỗ", "mùi da",
    # về xúc giác tích cực:
    "mềm mại", "mịn màng", "ấm áp", "êm ái", "thoáng mát", "sảng khoái", "dễ chịu", "thoải mái", "sảng khoái",
    "mượt mà", "nhẵn nhụi", "láng mịn", "êm dịu", "tươi mát", "dễ chịu", "thoáng đãng", "sảng khoái", "thoải mái",
    # về sự vật tích cực:
    "hoa", "cây", "núi", "biển", "mặt trời", "sao", "cầu vồng", "chim", "bướm", "cá", "hoa quả", "rau củ",
    "thực phẩm", "đồ uống", "món ăn", "bữa ăn", "nước", "không khí", "ánh sáng", "màu sắc", "âm nhạc", "tình yêu",
    "gia đình", "bạn bè", "tình bạn", "tình yêu", "niềm vui", "hạnh phúc", "sự sống", "sự thật", "công lý", "tự do",
    "bình yên", "an toàn", "hy vọng", "niềm tin", "sự thật", "sự thật", "sự thật", "sự thật", "sự thật", "sự thật",
    # về hành động tích cực:
    "giúp đỡ", "chia sẻ", "yêu thương", "chăm sóc", "bảo vệ", "cứu giúp", "che chở", "động viên", "khuyến khích",
    "khen ngợi", "tôn trọng", "biết ơn", "tha thứ", "bao dung", "độ lượng", "nhân ái", "từ bi", "lòng tốt", "nhân hậu",
    "vị tha", "dũng cảm", "kiên cường", "quyết đoán", "sáng suốt", "nhạy bén", "tư duy logic", "sáng tạo", "bền bỉ",
    "chuyên nghiệp", "có trách nhiệm", "tận tâm", "chăm chỉ", "cẩn thận", "có tổ chức", "có kỷ luật", "có tinh thần đồng đội",
    "hòa đồng", "thân thiện", "lạc quan", "tích cực", "hài hước", "thú vị", "thu hút", "quyến rũ", "lôi cuốn",
    # về trạng thái tích cực:
    "hạnh phúc", "vui vẻ", "thoải mái", "bình yên", "an toàn", "tự do", "yên tâm", "an tâm"
    # So sánh giá
    "rẻ", "hợp lý", "phải chăng", "tốt", "hời", "trễ", "muộn"
]

neutral_words = [
    "bình thường", "thường", "trung bình", "cũng được", "tạm", "tàm tạm", "không ngon không dở",
    "vừa phải", "ổn", "không tệ", "không tốt không xấu", "như vậy", "được", "chấp nhận được",
    "không có gì đặc biệt", "không có gì nổi bật", "không có gì ấn tượng", "không có gì đáng chú ý",
    "không có gì khác biệt", "không có gì thay đổi", "không có gì mới", "không có gì khác",
    "không có gì đáng kể", "không có gì quan trọng", "không có gì cần bàn cãi", "không có gì cần lo lắng",
    "không có gì cần thay đổi", "không có gì cần sửa chữa", "không có gì cần cải thiện",
    "không có gì cần thêm", "không có gì cần bớt", "không có gì cần thay thế", "không có gì cần bổ sung",
    "không có gì cần thay đổi", "không có gì cần điều chỉnh", "không có gì cần sửa đổi", "không có gì cần chỉnh sửa",
    "không có gì cần thêm", "không có gì cần bớt", "không có gì cần thay thế", "không có gì cần bổ sung"
]

food_words = [
    "đồ ăn", "thức ăn", "gia vị", "đồ uống", "ăn", "uống", "nước tương", "nước mắm", "dầu",
    "cơm", "bún", "phở", "mì", "bánh mì", "pizza", "hamburger", "sushi", "steak", "gà rán", "cá kho", "canh", "súp", "salad",
    "gỏi", "nem", "chả giò", "salad", "súp", "bánh tráng trộn",
    "kem", "bánh ngọt", "trái cây", "chè", "sữa chua",
    "nước ngọt", "nước ép", "sinh tố", "trà", "cà phê", "rượu", "bia",
    "bò", "gà", "lợn", "vịt", "cá", "tôm", "cua", "mực", "trứng",
    "cà chua", "hành", "tỏi", "ớt", "rau muống", "rau cải", "cà rốt", "khoai tây", "bí ngô",
    "muối", "đường", "tiêu", "bột ngọt", "hạt nêm", "nước mắm", "tương ớt", "dầu ăn", "giấm", "mù tạt",
    "gạo", "ngô", "đậu", "hạt tiêu", "hạt điều",
    "nấu ăn", "chế biến", "nấu nướng", "chiên", "xào", "luộc", "hấp", "nướng", "kho", "hầm",
    "hương vị", "vị", "ngon", "béo", "ngọt", "mặn", "cay", "chua", "đắng",
    "thực đơn", "menu", "món ăn", "suất ăn", "phần ăn",
    "nhà hàng", "quán ăn", "quán cafe", "quán bar", "bếp", "lò nướng",
    "ăn uống", "thưởng thức", "ẩm thực", "văn hóa ẩm thực", "món ăn", "món"
]

staff_words = [
    "nhân viên", "phục vụ", "nhân sự", "lao động", "cán bộ", "công nhân viên", "bảo vệ", "nhân viên phục vụ",
    "nhân viên bán hàng", "nhân viên lễ tân", "nhân viên kỹ thuật", "nhân viên quản lý", "nhân viên hỗ trợ",
    "nhân viên chăm sóc khách hàng", "nhân viên giao hàng", "nhân viên thu ngân", "nhân viên bảo vệ",
    "nhân viên dọn dẹp", "nhân viên bếp", "nhân viên pha chế", "giao hàng", "bếp", "bếp phụ", "quản lý", "thái độ", "kỹ năng", "tinh thần","giao tiếp"
    , "giải quyết", "phản hồi", "đào tạo"
]

price_words = [
    "giá", "giá cả", "chi phí", "tiền", "giá thành", "giá trị", "mức giá", "giá bán", "giá niêm yết", "giá khuyến mãi",
    "giá ưu đãi", "giá sỉ", "giá lẻ", "giá trị thực", "giá trị tương đối", "giá trị thị trường", "giá trị gia tăng"
    , "chi tiêu", "tiết kiệm", "tốn kém", "lợi nhuận", "lỗ", "thu nhập", "doanh thu", "chi phí sản xuất", "chi phí vận chuyển",
    "chi phí nhân công", "chi phí marketing", "chi phí bảo hành", "chi phí sửa chữa", "chi phí bảo trì", "chi phí đầu tư",
    "chi phí thuê", "chi phí nhiên liệu", "chi phí điện nước", "chi phí tài chính", "chi phí quản lý", "chi phí marketing"
]

store_words = [
    # Cụ thể cho cửa hàng đồ ăn
    "môi trường phục vụ", "tìm", "vị trí", "địa điểm", "cơ sở vật chất", "công nghệ hỗ trợ", "chỗ ngồi", "bầu không khí", "sự thoải mái",
   "môi trường", "bàn ghế", "nhà vệ sinh", "wc", "bếp", "bảo trì", "máy lạnh", "quạt", "khói"
]

deliver_words = [
    "giao hàng", "giao", 'grab', "be", "shopee", "tốc độ", "ship", "đúng hẹn", "đúng giờ", "địa chỉ"
]
# Function to clean and process text

# Function to clean and process text
emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"
                           u"\U0001F300-\U0001F5FF"
                           u"\U0001F680-\U0001F6FF"
                           u"\U0001F1E0-\U0001F1FF"
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           u"\U0001f926-\U0001f937"
                           u'\U00010000-\U0010ffff'
                           u"\u200d"
                           u"\u2640-\u2642"
                           u"\u2600-\u2B55"
                           u"\u23cf"
                           u"\u23e9"
                           u"\u231a"
                           u"\u3030"
                           u"\ufe0f"
                           "]+", flags=re.UNICODE)

def clean_text(text):
    text = text.lower()
    text = re.sub(emoji_pattern, " ", text)
    text = re.sub(r'([a-z]+?)\1+', r'\1', text)
    text = re.sub(r"(\w)\s*([" + string.punctuation + "])\s*(\w)", r"\1 \2 \3", text)
    text = re.sub(r"(\w)\s*([" + string.punctuation + "])", r"\1 \2", text)
    text = re.sub(f"([{string.punctuation}])([{string.punctuation}])+", r"\1", text)
    text = text.strip()
    while text.endswith(tuple(string.punctuation + string.whitespace)):
        text = text[:-1]
    while text.startswith(tuple(string.punctuation + string.whitespace)):
        text = text[1:]
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r"\s+", " ", text)
    text = word_tokenize(text, format="text")
    return text

# Function to find word counts and word lists
def find_words_count(document, list_of_words):
    document_lower = document.lower().replace("_", " ")
    word_count = 0
    for word in list_of_words:
        if word in document_lower:
            word_count += document_lower.count(word)
    return word_count

def find_words_list(document, list_of_words):
    document_lower = document.lower().replace("_", " ")
    word_list = []
    for word in list_of_words:
        if word in document_lower:
            word_list.append(word)
    return word_list

# Load data
uploaded_file_rs = pd.read_csv("data/df_res.csv")
df_rev1 = pd.read_csv("data/df_rev1.csv")
df_rev2 = pd.read_csv("data/df_rev2.csv")
df_rev = pd.concat([df_rev1, df_rev2])

file = open('data/files/vietnamese-stopwords.txt', 'r', encoding="utf8")
stopwords_lst = file.read().split('\n')
file.close()

df_PNR = pd.read_csv('DF_PNR.csv')

# Load the model
joblib_file = "logistic_regression_model.pkl"
loaded_model = joblib.load(joblib_file)
# Convert the text data to numerical vectors
vectorizer = TfidfVectorizer(analyzer='word', min_df=0, stop_words=stopwords_lst)
vectorizer.fit_transform(df_PNR['Comment'].tolist())

# Streamlit GUI
st.title("Data Science Project")
st.image("background_1.jpg")

# GUI Menu
menu = ["Trang chủ", "Thống kê sàn", "Thống kê cửa hàng", "Dự đoán ngữ nghĩa"]
choice = st.sidebar.selectbox('Menu', menu)

# Main logic based on menu choice
if choice == 'Trang chủ':
    st.subheader("Giới thiệu")
    st.write("Trang web hỗ trợ các tính năng sau: ")
    st.markdown("- Thống kê mô tả về tình hình hoạt động của sàn thương mại điện tử")
    st.markdown("- Thống kê mô tả về tình hình hoạt động của cửa hàng trên sàn thương mại điện tử")
    st.markdown("- Dự đoán ngữ nghĩa về bình luận của người dùng")
    st.markdown("---")

elif choice == "Thống kê sàn":
    st.subheader("Thống kê sàn")
    st.dataframe(uploaded_file_rs)
    
    st.title('Restaurant Distribution by District')
    plt.figure(figsize=(10, 6))
    sns.countplot(x='District', data=uploaded_file_rs)
    plt.title('Number of Restaurants in Each District')
    plt.xlabel('District')
    plt.ylabel('Number of Restaurants')
    st.pyplot(plt)

    st.title('Distribution of Average Price')
    plt.figure(figsize=(10, 6))
    sns.histplot(uploaded_file_rs['avg_price'], bins=20, kde=True)
    plt.title('Distribution of Average Price')
    plt.xlabel('Average Price')
    plt.ylabel('Frequency')
    st.pyplot(plt)

    st.title('Average Price of Each District')
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='District', y='avg_price', data=uploaded_file_rs)
    plt.title('Average Price of Each District')
    plt.xlabel('District')
    plt.ylabel('Average Price')
    st.pyplot(plt)

    st.title('Comment Type')
    totals = uploaded_file_rs[['negative_deliver', 'negative_food', 'negative_price', 'negative_staff', 'negative_store',
                               'neutral_deliver', 'neutral_food', 'neutral_price', 'neutral_staff', 'neutral_store',
                               'positive_deliver', 'positive_food', 'positive_price', 'positive_staff', 'positive_store']].sum()
    totals_df = totals.reset_index()
    totals_df.columns = ['Comment Type', 'Count']
    bars = alt.Chart(totals_df).mark_bar().encode(
        x=alt.X('Comment Type', sort=None, axis=alt.Axis(labelAngle=90)),
        y=alt.Y('Count', scale=alt.Scale(type='log'))
    ).properties(
        title='Comment type',
        width=600,
        height=400
    )
    text = bars.mark_text(
        align='center',
        baseline='bottom',
        dy=-5,
        color='white'
    ).encode(
        text='Count:Q'
    )
    chart = bars + text
    st.altair_chart(chart, use_container_width=True)

elif choice == 'Thống kê cửa hàng':
    st.subheader("Thống kê cửa hàng")
    st.dataframe(uploaded_file_rs)
    selected_res = st.text_input(label="Input ShopID: ")

    if selected_res:
        selected_res = int(selected_res)
        
        def show_res(selected_res):
            res = uploaded_file_rs[uploaded_file_rs['ID'] == selected_res]
            return res

        def show_rev(selected_res):
            rev = df_rev[df_rev['IDRestaurant'] == selected_res]
            return rev

        res = show_res(selected_res)
        rev = show_rev(selected_res)

        st.write("### Cửa hàng: ", res["Restaurant"].iloc[0])
        st.write("**Địa chỉ:** ", res["Address"].iloc[0])
        st.write("**Giá món:** ", res["Price"].iloc[0])
        st.write("**Đánh giá:** ", round(res["Rating"].iloc[0], 2))

        res_comt = res[['negative_deliver', 'negative_food', 'negative_price', 'negative_staff', 'negative_store',
                        'neutral_deliver', 'neutral_food', 'neutral_price', 'neutral_staff', 'neutral_store',
                        'positive_deliver', 'positive_food', 'positive_price', 'positive_staff', 'positive_store']].sum()

        totals_df = res_comt.reset_index()
        totals_df.columns = ['Comment Type', 'Count']
        st.write("### Review Data")
        st.dataframe(rev)
        
        bars = alt.Chart(totals_df).mark_bar().encode(
            x=alt.X('Comment Type', sort=None, axis=alt.Axis(labelAngle=90)),
            y=alt.Y('Count', scale=alt.Scale(type='linear')),
            color=alt.condition(
                alt.datum.Count > 0,
                alt.value('steelblue'),
                alt.value('lightgray')
            )
        ).properties(
            title='Comment type',
            width=600,
            height=400
        )
        text = bars.mark_text(
            align='center',
            baseline='middle',
            dy=-10,
            color='white'
        ).encode(
            text='Count:Q'
        )
        chart = bars + text
        st.altair_chart(chart, use_container_width=True)

        all_words = [token for token in rev['corpus'].tolist() if token and token != '']
        corpus = ' '.join(all_words)
        all_words_freq = nltk.FreqDist(all_words)

        negative_list = [sublist.replace("'", "").replace("[", "").replace("]", "").split(", ") for sublist in rev[rev['Rating'] <= 2]['corpus'].tolist()]
        negative_list = [item for sublist in negative_list for item in sublist if item and item != '']
        negative_corpus = ' '.join(negative_list)
        negative_freq = nltk.FreqDist(negative_list)

        st.write('**Wordcloud:**')
        word_cloud1 = WordCloud(max_words=500, background_color='white', scale=3).generate(corpus)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(word_cloud1)
        plt.axis('off')
        st.pyplot(fig)

        st.write('**Negative Wordcloud:**')
        word_cloud2 = WordCloud(max_words=500, background_color='white', scale=3).generate(negative_corpus)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(word_cloud2)
        plt.axis('off')
        st.pyplot(fig)

elif choice == 'Dự đoán ngữ nghĩa':
    st.subheader("Dự đoán ngữ nghĩa")
    selected_text = st.text_input(label="Nhập cảm nghĩ của quý khách: ")

    if selected_text:
        st.write("Kết quả dự đoán: ")
        new_comments = [selected_text]
        
        # Clean and transform the input text
        clean_comments = [clean_text(comment) for comment in new_comments]
        new_X = vectorizer.transform(clean_comments)
        predictions = loaded_model.predict(new_X)

        label_map = {1: 'Hài lòng', 0: 'Không hài lòng'}
        predicted_labels = label_map[predictions[0]]

        st.write(predicted_labels)


