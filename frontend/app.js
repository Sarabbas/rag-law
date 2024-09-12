document.addEventListener("DOMContentLoaded", function () {
    // Update this line with the correct WebSocket URL
    const websocketUrl = 'ws://localhost:7000/ws';
    const socket = new WebSocket(websocketUrl);

    const messagesContainer = document.getElementById("messages");
    const inputMessage = document.getElementById("inputMessage");
    const sendButton = document.getElementById("sendButton");

    let currentResponse = '';
    let lastSentMessage = '';
    let lastResponse = '';
    let isReceiving = false;

    const appendMessage = (message, from, isStreaming = false) => {
        let messageElement = document.getElementById('streaming-message');

        if (isStreaming) {
            if (!messageElement) {
                messageElement = document.createElement('div');
                messageElement.id = 'streaming-message';
                messageElement.classList.add('message');
                messagesContainer.appendChild(messageElement);
            }
            messageElement.innerHTML = `${from}: ${message.replace(/\n/g, '<br />')}`;
        } else {
            const finalMessageElement = document.createElement('div');
            finalMessageElement.classList.add('message');
            finalMessageElement.innerHTML = `${from}: ${message.replace(/\n/g, '<br />')}`;
            messagesContainer.appendChild(finalMessageElement);

            if (messageElement) {
                messageElement.remove();
            }
        }

        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    };

    socket.onopen = () => {
        console.log('WebSocket connection established.');
    };

    socket.onmessage = (event) => {
        const data = event.data;

        if (data === '\n') {
            lastResponse = currentResponse;
            appendMessage(lastResponse, 'Bot');
            currentResponse = '';
            isReceiving = false;
            sendButton.disabled = false;
        } else {
            currentResponse += data;
            appendMessage(currentResponse, 'Bot', true);
        }
    };

    socket.onclose = () => {
        console.log('WebSocket connection closed.');
    };

    const sendMessage = () => {
        if (!inputMessage.value.trim() || isReceiving) {
            return;
        }

        lastSentMessage = inputMessage.value.trim();
        appendMessage(lastSentMessage, 'User');
        currentResponse = '';
        lastResponse = '';
        isReceiving = true;
        sendButton.disabled = true;

        if (socket.readyState === WebSocket.OPEN) {
            socket.send(lastSentMessage);
        } else {
            console.log("WebSocket is not open. Message not sent.");
        }

        inputMessage.value = '';
    };

    sendButton.addEventListener("click", sendMessage);

    inputMessage.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
            event.preventDefault();
            sendMessage();
        }
    });
});