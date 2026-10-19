
document.addEventListener("DOMContentLoaded", function () {
    const chatWindow = document.getElementById("aiChatWindow");
    const openButton = document.getElementById("aiFloatButton");
    const closeButton = document.getElementById("aiCloseButton");
    const questionInput = document.getElementById("clinicalQuestion");
    const sendButton = document.getElementById("aiSendButton");
    const chatBody = document.getElementById("aiChatBody");

    if (!chatWindow || !openButton || !questionInput || !sendButton || !chatBody) {
        return;
    }

    function toggleAIChat() {
        const isOpen = chatWindow.classList.toggle("open");

        if (isOpen) {
            questionInput.focus();
        }
    }

    openButton.addEventListener("click", toggleAIChat);

    if (closeButton) {
        closeButton.addEventListener("click", function () {
            chatWindow.classList.remove("open");
        });
    }

    function addMessage(message, type) {
        const bubble = document.createElement("div");
        bubble.className = "ai-chat-message " + type;
        bubble.textContent = message;
        chatBody.appendChild(bubble);
        chatBody.scrollTop = chatBody.scrollHeight;
        return bubble;
    }

    async function askClinicalChat() {
        const question = questionInput.value.trim();

        if (!question) {
            return;
        }

        addMessage(question, "user");
        questionInput.value = "";

        const answerBubble = addMessage("Thinking...", "bot");
        sendButton.disabled = true;

        // Use result information only when this page provides it.
        let resultContext = {};

        const contextElement = document.getElementById("resultContextData");

        if (contextElement) {
            try {
                resultContext = JSON.parse(contextElement.textContent);
            } catch (error) {
                console.error("Could not read result context:", error);
            }
        }

        try {
            const response = await fetch("/clinical-result-chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    question: question,
                    result_context: resultContext
                })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.answer || "Request failed.");
            }

            answerBubble.textContent = data.answer || "No answer received.";

        } catch (error) {
            console.error("CerviAI chatbot error:", error);
            answerBubble.textContent =
                "Unable to connect. Please try again.";
        } finally {
            sendButton.disabled = false;
            questionInput.focus();
        }
    }

    sendButton.addEventListener("click", askClinicalChat);

    questionInput.addEventListener("keydown", function (event) {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            askClinicalChat();
        }
    });
});
