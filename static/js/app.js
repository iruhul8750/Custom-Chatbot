class Chatbox {
    constructor() {
        this.args = {
            openButton: document.querySelector('.chatbox__toggle-btn'),
            chatBox: document.querySelector('.chatbox__support'),
            sendButton: document.querySelector('.send__button'),
            inputField: document.querySelector('#chat-input'),
//            closeButton: document.querySelector('.chatbox__close-btn'),
            quickQuestions: document.querySelectorAll('.quick-question'),
            emojiToggle: document.querySelector('.emoji-toggle-btn'),
            emojiPicker: document.querySelector('.emoji-picker'),
            emojiOptions: document.querySelectorAll('.emoji-option')
        }
        this.state = false;
        this.messages = [];
        this.errorCount = 0;
        this.MAX_ERRORS = 3;
        this.hasNewMessage = false;
        this.emojiPickerVisible = false;
    }

    display() {
        const { openButton, chatBox, sendButton, inputField, quickQuestions, emojiToggle, emojiPicker, emojiOptions } = this.args;

        openButton.addEventListener('click', () => this.toggleState(chatBox));
//        closeButton.addEventListener('click', () => this.toggleState(chatBox));

        sendButton.addEventListener('click', () => this.onSendButton(chatBox));

        inputField.addEventListener('keypress', (e) => {
            if (e.key === "Enter") {
                this.onSendButton(chatBox);
            }
        });

        // Emoji picker functionality
        emojiToggle.addEventListener('click', (e) => {
            e.stopPropagation();
            this.toggleEmojiPicker();
        });

        emojiOptions.forEach(emoji => {
            emoji.addEventListener('click', () => {
                this.insertEmoji(emoji.textContent);
            });
        });

        document.addEventListener('click', () => {
            if (this.emojiPickerVisible) {
                this.hideEmojiPicker();
            }
        });

        emojiPicker.addEventListener('click', (e) => {
            e.stopPropagation();
        });

        quickQuestions.forEach(question => {
            question.addEventListener('click', () => {
                const text = question.getAttribute('data-question');
                this.args.inputField.value = text;
                this.onSendButton(chatBox);
            });
        });

        // Add initial welcome message
        this.addWelcomeMessage(chatBox);
    }

    toggleEmojiPicker() {
        this.emojiPickerVisible = !this.emojiPickerVisible;
        if (this.emojiPickerVisible) {
            this.args.emojiPicker.classList.add('show');
        } else {
            this.args.emojiPicker.classList.remove('show');
        }
    }

    hideEmojiPicker() {
        this.emojiPickerVisible = false;
        this.args.emojiPicker.classList.remove('show');
    }

    insertEmoji(emoji) {
        const inputField = this.args.inputField;
        const startPos = inputField.selectionStart;
        const endPos = inputField.selectionEnd;
        const currentValue = inputField.value;

        inputField.value = currentValue.substring(0, startPos) + emoji + currentValue.substring(endPos);
        inputField.focus();
        inputField.selectionStart = startPos + emoji.length;
        inputField.selectionEnd = startPos + emoji.length;

        this.hideEmojiPicker();
    }

    addWelcomeMessage(chatbox) {
        const welcomeMessage = {
            name: "Bagiya",
            message: "Hello! 👋 I'm the Bagiya School assistant. How can I help you today?",
            type: 'visitor'
        };
        this.messages.push(welcomeMessage);
        this.updateChatText(chatbox);
        this.showNotificationBadge();
    }

    toggleState(chatbox) {
        this.state = !this.state;

        if (this.state) {
            chatbox.classList.add('chatbox--active');
            this.hideNotificationBadge();
            this.hideEmojiPicker();
        } else {
            chatbox.classList.remove('chatbox--active');
        }
    }

    showNotificationBadge() {
        if (!this.state) {
            this.hasNewMessage = true;
            document.querySelector('.chatbox__badge').style.display = 'flex';
        }
    }

    hideNotificationBadge() {
        this.hasNewMessage = false;
        document.querySelector('.chatbox__badge').style.display = 'none';
    }

    onSendButton(chatbox) {
        const textField = this.args.inputField;
        const text1 = textField.value.trim();
        if (text1 === "") {
            return;
        }

        let msg1 = { name: "You", message: text1, type: 'operator' };
        this.addMessage(chatbox, msg1.name, msg1.message, msg1.type);
        textField.value = '';

        this.showTyping(chatbox);

        fetch('/predict', {
            method: 'POST',
            body: JSON.stringify({ message: text1 }),
            headers: {
                'Content-Type': 'application/json'
            },
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(response => {
            this.removeTyping();
            this.errorCount = 0;

            let message = response.answer || "I didn't understand that. Could you rephrase?";
            let msgType = response.model_used === 'error' ? 'error' : 'visitor';

            this.addMessage(chatbox, "Bagiya", message, msgType);
            if (!this.state) {
                this.showNotificationBadge();
            }
        })
        .catch((error) => {
            console.error('Error:', error);
            this.removeTyping();
            this.errorCount++;

            let errorMsg = "Sorry, I'm having trouble connecting.";
            if (this.errorCount >= this.MAX_ERRORS) {
                errorMsg += " Please try again later.";
                this.args.sendButton.disabled = true;
                setTimeout(() => {
                    this.args.sendButton.disabled = false;
                    this.errorCount = 0;
                }, 30000);
            }

            this.addMessage(chatbox, "Bagiya", errorMsg, 'error');
            if (!this.state) {
                this.showNotificationBadge();
            }
        });
    }

    addMessage(chatbox, name, message, type) {
        this.messages.push({ name, message, type });
        this.updateChatText(chatbox);
    }

    showTyping(chatbox) {
        const messagesContainer = chatbox.querySelector('.chatbox__messages');
        const typingElement = document.createElement('div');
        typingElement.className = 'messages__item messages__item--visitor typing-indicator';
        typingElement.id = 'typing-indicator';
        typingElement.innerHTML = '<span></span><span></span><span></span>';
        messagesContainer.appendChild(typingElement);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    removeTyping() {
        const typingElement = document.getElementById('typing-indicator');
        if (typingElement) {
            typingElement.remove();
        }
    }

    updateChatText(chatbox) {
        let html = '';
        this.messages.slice(-20).forEach((item) => {
            const messageClass = item.type === 'operator' ?
                'messages__item--operator' :
                (item.type === 'error' ? 'messages__item--error' : 'messages__item--visitor');

            html += `<div class="messages__item ${messageClass}">${item.message}</div>`;
        });

        const chatmessage = chatbox.querySelector('.chatbox__messages');
        chatmessage.innerHTML = html;
        chatmessage.scrollTop = chatmessage.scrollHeight;
    }
}

const chatbox = new Chatbox();
chatbox.display();