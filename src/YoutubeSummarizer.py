import os
import yt_dlp
import whisper
import re  # Import for cleaning text
import textwrap
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS

# Load environment variables
load_dotenv()
embeddings = OpenAIEmbeddings()

# Initialize OpenAI Model
llm = ChatOpenAI(model_name="gpt-4", temperature=0)

# Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,  # Slight overlap to maintain context
    separators=["\n\n", ".", ",", " "]
)

def get_video_info(url):
    filename = os.path.join(os.getcwd(), "lecuninterview.%(ext)s")  # Dynamic extension
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': filename,
        'quiet': False,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',  # Change to 'mp4' if needed
        }]
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            result = ydl.extract_info(url, download=True)
            return result
        except Exception as e:
            print(f"Error downloading video: {e}")
            return None

url = "https://www.youtube.com/watch?v=mBjPyte2ZZo"
video_info = get_video_info(url)

if video_info:
    print("Checking downloaded files...")
    downloaded_file = None

    for file in os.listdir():
        if file.startswith("lecuninterview"):
            downloaded_file = file
            print("Found file:", downloaded_file)
            break

    if not downloaded_file:
        print("Error: No downloaded file found!")
    else:
        # Load Whisper model
        model = whisper.load_model("tiny")

        # Transcribe the correct file
        try:
            print("Transcribing...")
            result = model.transcribe(downloaded_file)

            # Extract only text from Whisper's result
            if "segments" in result:
                transcribed_text = " ".join([segment["text"] for segment in result["segments"]])
            else:
                transcribed_text = result.get("text", "").strip()

            # Remove numbers, timestamps, and unwanted symbols
            transcribed_text = re.sub(r'\d+:\d+(:\d+)?', '', transcribed_text)  # Removes timestamps
            transcribed_text = re.sub(r'\s+', ' ', transcribed_text).strip()  # Remove extra spaces

            # Save to file
            with open('text.txt', 'w') as f:
                f.write(transcribed_text)

            # Read cleaned transcribed text
            with open('text.txt', 'r') as f:
                text = f.read()

            # Split text into chunks
            texts = text_splitter.split_text(text)
            docs = [Document(page_content=t) for t in texts]

            # Store in FAISS
            faiss_store = FAISS.from_texts([doc.page_content for doc in docs], embeddings)

            print("Transcript stored. You can now ask questions.")

            # Choose best summarization method
            if len(text) < 3000:
                chain_type = "stuff"
            elif len(text) < 10000:
                chain_type = "refine"
            else:
                chain_type = "map_reduce"

            # Load summarization chain
            chain = load_summarize_chain(llm, chain_type=chain_type)
            output_summary = chain.invoke(docs)

            # Ensure clean formatting before displaying
            wrapped_text = textwrap.fill(str(output_summary), width=100)
            #print("\nFinal Summary:\n", wrapped_text)

            # ✅ Call ask_question() after storing transcript
            def ask_question(query):
                retrieved_docs = faiss_store.similarity_search(query, k=3)
                context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                #print("\n📜 Retrieved Context:\n", context)

                response = llm.invoke(f"Based on the following text, answer the question:\n\n{context}\n\nQuestion: {query}")
                #print("\n🤖 Response:\n", response)

            # ✅ Ask a sample question
            user_question = "What did the speaker say about the future of AI?"
            ask_question(user_question)

        except Exception as e:
            print(f"Error during transcription: {e}")

else:
    print("Failed to download video.")
