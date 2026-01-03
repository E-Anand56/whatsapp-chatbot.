import streamlit as st
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="WhatsApp Chat Analyzer",
    layout="centered"
)

st.title("📱 WhatsApp Chat Analyzer")
st.caption("Sentiment • Flirt Detection • Activity Analysis")

# ----------------------------
# LOAD DATA
# ----------------------------
@st.cache_data
def load_data():
    data = pd.read_csv(
        "WhatsApp Chat with Selva pullingow.txt",
        header=None,
        on_bad_lines='skip',
        encoding='utf-8-sig'
    )

    data = data.drop([0, 1, 2], errors="ignore")
    data = data.drop(columns=[2, 3], errors="ignore")
    data.columns = ['Date', 'Message']

    data[['Time', 'Rest']] = data['Message'].str.split(' - ', n=1, expand=True)
    data[['Name', 'chat']] = data['Rest'].str.split(': ', n=1, expand=True)
    data.drop(columns=['Message', 'Rest'], inplace=True)

    data = data[data['chat'].notnull()]
    return data

data = load_data()

# ----------------------------
# SENTIMENT ANALYZER
# ----------------------------
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = analyzer.polarity_scores(text)['compound']
    if score > 0.05:
        return "Positive 😊"
    elif score < -0.05:
        return "Negative 😠"
    else:
        return "Neutral 😐"

def get_flirt(text):
    emojis = ['😍','😘','❤️','😂','🤣','😉','😜']
    return "Flirt 💕" if any(e in text for e in emojis) else "Normal 💬"

# ----------------------------
# SIDEBAR – USER STATS
# ----------------------------
st.sidebar.header("👤 User Statistics")

users = data['Name'].dropna().unique().tolist()
selected_user = st.sidebar.selectbox("Select Person", users)

user_data = data[data['Name'] == selected_user]

total_messages = user_data.shape[0]
media_count = user_data['chat'].str.contains('Media', na=False).sum()

st.sidebar.metric("💬 Total Messages", total_messages)
st.sidebar.metric("📎 Media Sent", media_count)

# ----------------------------
# CHAT ANALYZER UI
# ----------------------------
st.subheader("💡 Message Analyzer")

user_input = st.text_area(
    "Enter a WhatsApp message:",
    placeholder="Type your message here..."
)

if st.button("Analyze Message"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        sentiment = get_sentiment(user_input)
        flirt = get_flirt(user_input)

        col1, col2 = st.columns(2)

        col1.success(f"Sentiment: {sentiment}")
        col2.info(f"Flirt Prediction: {flirt}")

# ----------------------------
# TALKATIVE ANALYSIS
# ----------------------------
st.subheader("📊 Talkative Analysis")

msg_counts = data['Name'].value_counts()
talkative = msg_counts.idxmax()
less_talkative = msg_counts.idxmin()

st.write(f"🗣️ **Most Talkative:** {talkative} ({msg_counts[talkative]} messages)")
st.write(f"🤫 **Less Talkative:** {less_talkative} ({msg_counts[less_talkative]} messages)")

# ----------------------------
# MOST ACTIVE DAY & HOUR
# ----------------------------
st.subheader("⏰ Activity Insights")

data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data['Day'] = data['Date'].dt.day_name()

data['Hour'] = pd.to_datetime(
    data['Time'].str.replace('\u202f', ' '),
    format='mixed',
    errors='coerce'
).dt.hour

st.write("📅 **Most Active Day:**", data['Day'].value_counts().idxmax())
st.write("🕒 **Most Active Hour:**", f"{data['Hour'].value_counts().idxmax()}:00")

# ----------------------------
# SENTIMENT SUMMARY TABLE
# ----------------------------
st.subheader("📈 Sentiment Summary")

sentiment_table = (
    data.groupby(['Name', 'Sentiment'])
        .size()
        .unstack(fill_value=0)
)

st.dataframe(sentiment_table, use_container_width=True)
