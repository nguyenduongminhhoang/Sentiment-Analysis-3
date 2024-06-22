import numpy as np
import pandas as pd
import re
import string
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from wordcloud import WordCloud
import nltk
from underthesea import word_tokenize, text_normalize
import squarify
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
vectorizer = TfidfVectorizer(analyzer='word', min_df=0.0, stop_words=stopwords_lst)
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
# ---------------------------
# function to classify the content of comment into neutral, positive, negative
def find_words_count(document, list_of_words):
    document_lower = document.lower().replace("_"," ")
    word_count = 0
    word_list = []

    for word in list_of_words:
        if word in document_lower:
            word_count += document_lower.count(word)
            word_list.append(word)
    return word_count

def find_words_list(document, list_of_words):
    document_lower = document.lower().replace("_"," ")
    word_count = 0
    word_list = []

    for word in list_of_words:
        if word in document_lower:
            word_count += document_lower.count(word)
            word_list.append(word)
    return word_list
# ---------------------------
# function to process text
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
    "]+", flags=re.UNICODE) # Unicode emojis.

def clean_text(text):
    text = text.lower() # lowercase text

    text = re.sub(emoji_pattern, " ", text) # remove emojis

    text = re.sub(r'([a-z]+?)\1+',r'\1', text) # reduce repeated character (e.g. 'aaabbb' -> 'ab')

    # Ensure space before and after any punctuation mark
    text = re.sub(r"(\w)\s*([" + string.punctuation + "])\s*(\w)", r"\1 \2 \3", text)
    text = re.sub(r"(\w)\s*([" + string.punctuation + "])", r"\1 \2", text)

    text = re.sub(f"([{string.punctuation}])([{string.punctuation}])+",r"\1", text) # reduce consecutive punctuation

    # Remove any leading or trailing spaces, or leading or trailing punctuation marks from the text
    text = text.strip()
    while text.endswith(tuple(string.punctuation+string.whitespace)):
        text = text[:-1]
    while text.startswith(tuple(string.punctuation+string.whitespace)):
        text = text[1:]

    text = text.translate(str.maketrans('', '', string.punctuation)) # remove all punctuation

    text = re.sub(r"\s+", " ", text) # reduce multiple spaces

    text = text_normalize(text) # make sure punctunation is in the right letter (Vietnamese case)
    text = word_tokenize(text, format="text") # tokenize the cleaned text
    # text = unidecode(text) # remove accent marks from sentences (no significant difference when accent marks is removed or kept)
    return text

# ---------------------------
# GUI
st.title("Data Science Project")
st.image("background_1.jpg")

# ---------------------------
# GUI Menu
menu = ["Trang chủ", "Thống kê sàn", "Thống kê cửa hàng", "Dự đoán ngữ nghĩa"]
choice = st.sidebar.selectbox('Menu', menu)

# ---------------------------
if choice == 'Trang chủ':    
    st.subheader("Giới thiệu")
    st.write("Trang web hỗ trợ các tính năng sau: ")
    st.markdown("- Thống kê mô tả về tình hình hoạt động của sàn thương mại điện tử")
    st.markdown("- Thống kê mô tả về tình hình hoạt động của cửa hàng trên sàn thương mại điện tử")
    st.markdown("- Dự đoán ngữ nghĩa về bình luận của người dùng")
    st.markdown("---")

elif choice == "Thống kê sàn":
    st.subheader("Thống kê sàn")
    df_rev = None
    df_res = None
    df_res = uploaded_file_rs #pd.read_csv(uploaded_file_rs)
    # global lines
    st.dataframe(df_res)
    # -------------------------------------
    st.title('Restaurant Distribution by District')
    # Vẽ biểu đồ bằng matplotlib và seaborn
    plt.figure(figsize=(10, 6))
    sns.countplot(x='District', data=df_res)
    plt.title('Number of Restaurant in Each District')
    plt.xlabel('District')
    plt.ylabel('Number of Restaurant')
    # Hiển thị biểu đồ trong Streamlit
    st.pyplot(plt)
    #-------------------------------------
    # Thiết lập giao diện Streamlit
    st.title('Distribution of Average Price')
    # Vẽ biểu đồ histogram bằng matplotlib và seaborn
    plt.figure(figsize=(10, 6))
    sns.histplot(df_res['avg_price'], bins=20, kde=True)
    plt.title('Distribution of Average Price')
    plt.xlabel('Average Price')
    plt.ylabel('Frequency')
    # Hiển thị biểu đồ trong Streamlit
    st.pyplot(plt)
    #-------------------------------------
    # Thiết lập giao diện Streamlit
    st.title('Average Price of Each District')
    # Vẽ biểu đồ boxplot bằng matplotlib và seaborn
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='District', y='avg_price', data=df_res)
    plt.title('Average Price of Each District')
    plt.xlabel('District')
    plt.ylabel('Average Price')
    # Hiển thị biểu đồ trong Streamlit
    st.pyplot(plt)
    #-------------------------------------
    # Tính tổng số lượng comment ở mỗi cột
    st.title('Comment Type')
    totals = df_res[['negative_deliver', 'negative_food', 'negative_price', 'negative_staff', 'negative_store',
                    'neutral_deliver', 'neutral_food', 'neutral_price', 'neutral_staff', 'neutral_store',
                    'positive_deliver', 'positive_food', 'positive_price', 'positive_staff', 'positive_store']].sum()
    totals_df = totals.reset_index()
    totals_df.columns = ['Comment Type', 'Count']
    # Tạo biểu đồ cột bằng altair với tỷ lệ logarit
    bars = alt.Chart(totals_df).mark_bar().encode(
        x=alt.X('Comment Type', sort=None, axis=alt.Axis(labelAngle=90)),
        y=alt.Y('Count', scale=alt.Scale(type='log'))
    ).properties(
        title='Comment type',
        width=600,
        height=400
    )
    # Thêm nhãn giá trị vào các thanh
    text = bars.mark_text(
        align='center',
        baseline='bottom',
        dy=-5,  # Move text slightly above the bar
        color='white'
    ).encode(
        text='Count:Q'
    )
    chart = bars + text
    # Hiển thị biểu đồ trong Streamlit
    st.altair_chart(chart, use_container_width=True)

elif choice == 'Thống kê cửa hàng':
    st.subheader("Thống kê cửa hàng")
    df_res = None
    df_res = uploaded_file_rs
    st.dataframe(df_res)
    selected_res = st.text_input(label="Input ShopID: ")
    if selected_res: 
        # selected_res = int(selected_res)
        def show_res(selected_res):
            res = df_res[df_res['ID']==selected_res]
            return res
        def show_rev(selected_res):
            rev = df_rev[df_rev['IDRestaurant']== selected_res]
            return rev
        res = show_res(int(selected_res))
        rev = show_rev(int(selected_res))
        # Hiển thị thông tin nhà hàng
        st.write("### Cửa hàng: ", res["Restaurant"].iloc[0])
        st.write("**Địa chỉ:** ", res["Address"].iloc[0])
        st.write("**Giá món:** ", res["Price"].iloc[0])
        st.write("**Đánh giá:** ", round(res["Rating"].iloc[0],2))
        # Tính tổng số lượng comment ở mỗi cột
        res_comt = res[['negative_deliver', 'negative_food', 'negative_price', 'negative_staff', 'negative_store',
                        'neutral_deliver', 'neutral_food', 'neutral_price', 'neutral_staff', 'neutral_store',
                        'positive_deliver', 'positive_food', 'positive_price', 'positive_staff', 'positive_store']].sum()

        # Tạo DataFrame mới từ tổng số lượng comment
        totals_df = res_comt.reset_index()
        totals_df.columns = ['Comment Type', 'Count']
        
        st.write("### Review Data")
        st.dataframe(rev)
        # Tạo biểu đồ cột bằng altair với tỷ lệ logarit
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

        # Thêm nhãn giá trị vào các thanh với màu trắng
        text = bars.mark_text(
            align='center',
            baseline='middle',
            dy=-10,  # Di chuyển nhãn lên trên một chút
            color='white'  # Màu chữ trắng
        ).encode(
            text='Count:Q'
        )

        # Kết hợp biểu đồ và nhãn
        chart = bars + text

        # Hiển thị biểu đồ trong Streamlit
        st.altair_chart(chart, use_container_width=True)

        # Tạo WORD CLOUD của corpus
        all_words = [token for token in rev['corpus'].tolist() if token and token != '']

        corpus = ' '.join(all_words)
        all_words_freq = nltk.FreqDist(all_words)

        negative_list = []
        for sublist in rev['negative_list']:
            negative_list.append(sublist.replace("'", "").replace("[", "").replace("]", ""))
        
        negative_text = ' '.join(negative_list)

        p_list = []
        for sublist in rev['positive_list']:
            p_list.append(sublist.replace("'", "").replace("[", "").replace("]", ""))
        
        p_text = ' '.join(p_list)
        st.markdown("---")
        # Print the total number of words and the 15 most common words
        st.write('### Wordcloud')
        st.write('**Number of reviews: {}**'.format(len(all_words_freq)))

        # Tạo và hiển thị Word Cloud
        st.write('**Positive Wordcloud:**')
        word_cloud1 = WordCloud(max_words=100, background_color = 'white',
                            width=2000, height=1000).generate(p_text)
        # Hiển thị Word Cloud
        plt.figure(figsize=(12, 8))
        plt.axis("off")
        plt.imshow(word_cloud1, interpolation='bilinear')
        st.pyplot(plt)


        # Tạo và hiển thị Word Cloud
        st.write('**Negative Wordcloud:**')
        word_cloud = WordCloud(max_words=100, background_color = 'white',
                            width=2000, height=1000).generate(negative_text)
        

        # Hiển thị Word Cloud
        plt.figure(figsize=(12, 8))
        plt.axis("off")
        plt.imshow(word_cloud, interpolation='bilinear')
        st.pyplot(plt)
        
        
        
        def pnr_level(df):
            if (df['Positive'] == 0) & (df['Negative'] == 0):
                return 'Unidentified'
            
            elif df['PNR_Score'] >= 10:
                return 'Very satisfied'
            
            elif df['P'] >= 4 and df['N'] >= 4 and df['R'] <= 2:
                return 'Contradict Negative'
            
            elif df['P'] <= 3 and df['N'] <= 3 and df['R'] >= 3:
                return 'Contradict Positive'

            elif df['PNR_Score'] <= 7:
                return 'Not satisfied'
            
            else:
                return 'Normal'
            
        # Score range
        po_labels = range(1, 6)
        ne_labels = range(5, 0, -1)
        r_labels = range(1, 6)

        mask = (rev['positive_count'] == 0) & (rev['negative_count'] == 0)
        df_pnr = rev[~mask]
        # Check for NaN values
        if df_pnr['positive_count'].isnull().any():
            st.write("### Error!")
            st.write("NaN values found in 'positive_count' column")

        # Check the number of unique values
        unique_values_p = df_pnr['positive_count'].nunique()
        unique_values_n = df_pnr['positive_count'].nunique()
        if unique_values_p < 5:
            st.write("### Warning!")
            st.write(f"Không đủ dữ liệu để thực hiện phân cụm khách hàng!")

        elif unique_values_n < 5:
            st.write("### Warning!")
            st.write(f"Không đủ dữ liệu để thực hiện phân cụm khách hàng!")


        else:
            # Assign these labels to 4 equal percentile groups
            p_groups = pd.qcut(df_pnr['positive_count'].rank(method='first'), q=5, labels=po_labels)

            n_groups = pd.qcut(df_pnr['negative_count'].rank(method='first'), q=5, labels=ne_labels)

            r_groups = pd.qcut(df_pnr['Rating'].rank(method='first'), q=5, labels=r_labels)

            df_pnr = df_pnr.assign(P = p_groups.values, N = n_groups.values,  R = r_groups.values)

            # Join the score
            def join_rfm(x): return str(int(x['P'])) + str(int(x['N'])) + str(int(x['R']))
            df_pnr['PNR_Segment'] = df_pnr.apply(join_rfm, axis=1)
            df_pnr['PNR_Score'] = df_pnr[['P','N','R']].sum(axis=1)


            def pnr_level(df):
                # Check for special 'STARS' and 'NEW' conditions first
                if (df['positive_count'] == 0) & (df['negative_count'] == 0):
                    return 'Unidentified'
                
                elif df['PNR_Score'] >= 10:
                    return 'Very satisfied'
                
                elif df['P'] >= 4 and df['N'] >= 4 and df['R'] <= 2:
                    return 'Contradict Negative'
                
                elif df['P'] <= 3 and df['N'] <= 3 and df['R'] >= 3:
                    return 'Contradict Positive'

                elif df['PNR_Score'] <= 7:
                    return 'Not satisfied'
                
                else:
                    return 'Normal'
                
            df_pnr['PNR_Level'] = df_pnr.apply(pnr_level, axis=1)
            df_pnr['PNR_Level'].value_counts()

            pnr_agg = df_pnr.groupby('PNR_Level').agg({
                'positive_count': 'mean',
                'negative_count': 'mean',
                'Rating': ['mean', 'count']
            })
            pnr_agg[('positive_count', 'mean')] = pnr_agg[('positive_count', 'mean')].round(0)
            pnr_agg[('negative_count', 'mean')] = pnr_agg[('negative_count', 'mean')].round(0)

            pnr_agg.columns = pnr_agg.columns.droplevel()
            pnr_agg.columns = ['PositiveMean','NegativeMean','RatingMean', 'Count']
            pnr_agg['Percent'] = round((pnr_agg['Count']/pnr_agg.Count.sum())*100, 2)

            # Reset the index
            pnr_agg = pnr_agg.reset_index()
            st.markdown("---")
            st.write('### Customer Segmentation:')
            st.write("**Very satisfied: Khách hàng thích** ")
            st.write("**Not satisfied: Khách hàng ghét** ")
            st.write("**Neutral: Bình thường** ")
            st.write("**Contradict positive: Khen ít chê nhiều nhưng điểm cao** ")
            st.write("**Contradict negative: Khen nhiều chê ít nhưng điểm thấp** ")


            plt.clf()
            fig1 = plt.gcf()
            ax = fig1.add_subplot()
            fig1.set_size_inches(15, 10)

            colors_dict = {0: 'yellow', 1: 'royalblue', 2: 'cyan', 3: 'pink', 4: 'green', 5: 'red'}

            squarify.plot(sizes=pnr_agg['Count'],
                        text_kwargs={'fontsize': 12, 'weight': 'bold', 'fontname': "sans serif"},
                        color=colors_dict.values(),
                        label=['{} \n{:.0f} positives \n{:.0f} negatives \n rating {:.2f} \n{:.0f} ({}%)'.format(*pnr_agg.iloc[i])
                                for i in range(0, len(pnr_agg))], alpha=0.5)

            plt.title("Reviewers Segments", fontsize=26, fontweight="bold")
            plt.axis('off')

            # Display the plot
            st.pyplot(fig1)
            



elif choice == 'Dự đoán ngữ nghĩa':
    st.subheader("Dự đoán ngữ nghĩa")
    selected_text = st.text_input(label = "Nhập cảm nghĩ của quý khách: ")
    if selected_text:
        
        st.write("Kết quả dự đoán: ")
        # Example of making a prediction
        new_comments = [selected_text]
        new_X = vectorizer.transform(new_comments)
        predictions = loaded_model.predict(new_X)

        # Map numerical labels back to sentiment labels
        label_map = {1: 'Hài lòng', 0: 'Không hài lòng'}
        predicted_labels = label_map[predictions[0]]
        st.write(predicted_labels)
