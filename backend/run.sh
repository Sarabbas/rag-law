docker build . -t rag-back
docker run -p 7000:7000 -v ./backend:/app rag-back