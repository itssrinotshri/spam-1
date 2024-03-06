import pickle
import streamlit as st

model = pickle.load(open("spam.pkl", "rb"))
cv = pickle.load(open("vectorizer.pkl", "rb"))

def predict_spam_or_ham(msg):
    vect = cv.transform([msg]).toarray()
    prediction = model.predict(vect)
    confidence = model.predict_proba(vect).max() * 100
    return prediction[0], confidence

def main():
    st.title("Email Spam Verifier")
    
    # Retrieve or initialize the message from session state
    msg = st.session_state.get("msg", "")

    # Text area for entering email text
    msg = st.text_area("Enter your email text here:", value=msg, height=150, key="email_text_area")

    # Store the message in session state
    st.session_state.msg = msg
    
    # Predict button
    if st.button("Predict"):
        if msg.strip() == "":
            st.error("Please enter some text before predicting.")
        else:
            prediction, confidence = predict_spam_or_ham(msg)
            if prediction == 1:
                st.error(f"This is a SPAM email with {confidence:.2f}% confidence.")
            else:
                st.success(f"This is a HAM email with {confidence:.2f}% confidence.")
    
    # Reset button
    if st.button("Reset"):
        # Clear the message in session state
        st.session_state.msg = ""
        # Clear the text area
        st.text_area("Enter your email text here:", value="", height=150, key="reset_email_text_area")
        st.empty()

    st.sidebar.subheader("About")
    st.sidebar.info(
        "This application uses a trained machine learning model to predict whether an email is spam or ham."
    )
    st.sidebar.subheader("Instructions")
    st.sidebar.markdown(
        "- Enter the text of your email in the text area.\n"
        "- Click the 'Predict' button to see the prediction.\n"
        "- The result will be displayed as either SPAM or HAM.\n"
        "- You can also reset the input."
    )

if __name__ == "__main__":
    main()
