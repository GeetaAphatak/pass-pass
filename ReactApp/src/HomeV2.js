// Page with sidebar, searchbox at the bottom and display the response in a scrollable container.
import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import rehypeRaw from 'rehype-raw';
import { assets } from './assets/assets';
import "./Home.css";
import 'ag-grid-community/styles/ag-theme-alpine.css';

// Define initial messages outside the component to reuse in reset
const initialMessages = [
  { type: 'bot', text: "Welcome! Tell me what you're interested in, and I'll guide you to the right application for your needs." },
  { type: 'bot', text: "Feel free to explore our options listed on the side." }
];

const Home = () => {
  const [messages, setMessages] = useState(initialMessages); 
  const [userInput, setUserInput] = useState('');
  const [cards, setCards] = useState([]);
  const [history, setHistory] = useState([]);  // State to track conversation history
  const chatContainerRef = useRef(null);

  useEffect(() => {
    // Load cards initially
    fetch('http://localhost:8889/config')
      .then(response => response.json())
      .then(data => setCards(data))
      .catch(error => console.error('Error fetching data:', error));
  }, []);

  const handleSendMessage = async () => {
    if (userInput.trim()) {
      const userMessage = { type: 'user', text: userInput };
      setMessages((prevMessages) => [...prevMessages, userMessage]);
      setHistory((prevHistory) => [...prevHistory, { "role": "user", "content": userInput }]);  // Update history with user input

      try {
        const response = await axios.post("http://localhost:8889/find_app", {
          user_query: userInput,
          history: history  // Send the entire conversation history
        });

        const { result } = response.data;
        if (result) {
          const convertedText = convertMarkdownLinksToHtml(result);
          const botMessage = { type: 'bot', text: convertedText };

          setMessages((prevMessages) => [...prevMessages, botMessage]);
          setHistory((prevHistory) => [...prevHistory, { "role": "assistant", "content": result }]);  // Update history with bot response
        } else {
          setMessages((prevMessages) => [
            ...prevMessages,
            { type: 'bot', text: "An unexpected error occurred." }
          ]);
        }
      } catch (error) {
        console.error("Error sending message:", error);
        setMessages((prevMessages) => [
          ...prevMessages,
          { type: 'bot', text: "Error occurred during processing." }
        ]);
      }

      setUserInput('');
    }
  };

  const convertMarkdownLinksToHtml = (text) => {
    const markdownLinkPattern = /\[([^\]]+)\]\((https?:\/\/[^\)]+)\)/g;
    return text.replace(markdownLinkPattern, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  };

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages]);

  return (
    <div className="container">
      <header className="header">
        <img src={assets.broadridge_logo} alt="Broadridge Logo" className="logo" />
        <center><h1 className="heading">Onboarding AI</h1></center> 
        <div className="icons">
          <i id="home" className="fa fa-home" style={{ fontSize: '30px' }}></i>
        </div>
      </header>

      <div className="main-content">
        <div className="content-wrapper">
          <div className="sidebar" id="sidebar">
            {cards.map(card => (
              <div key={card.name} className="card">
                <a href={card.link} className="card-link" target="_blank" rel="noopener noreferrer">
                  <div className="card-title">{card.name}</div>
                  <div className="card-desc">{card.short_description}</div>
                </a>
              </div>
            ))}
          </div>
          <div className="content-wrapper-chat">
            <div className="chat-container" ref={chatContainerRef}>
              <div className="chat-window">
                {messages.map((message, index) => (
                  <div key={index}>
                    {message.type === 'user' ? (
                      <div className="user-message">
                        <div className="message-bubble user-bubble">
                          <img src={assets.user_icon} alt="User" className="user-img" />
                          <span style={{ marginLeft: '10px' }}>{message.text}</span>
                        </div>
                      </div>
                    ) : (
                      <div className="response-message">
                        <div className="message-bubble bot-bubble">
                          <img src={assets.bot} alt="Robo" className="user-img" />
                          <div className="bot-response-text">
                            <ReactMarkdown rehypePlugins={[rehypeRaw]}>{message.text}</ReactMarkdown>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
              <div className="footer">
                <div className="search-box">
                  <input
                    type="text"
                    value={userInput}
                    onChange={(e) => {
                      setUserInput(e.target.value);
                    }}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault(); // Prevent default Enter behavior for form submission
                        if (userInput.trim()) handleSendMessage();
                      }
                    }}
                    placeholder="Hello! How can I assist you today?"
                  />
                  <img
                    src={assets.send_icon}
                    alt="Send"
                    className="btn mic"
                    onClick={handleSendMessage}
                  />
                </div>

                <div className="reset-btn">
                  <button onClick={() => {
                    setMessages(initialMessages); // Reset messages to initial state
                    setHistory([]); // Clear history
                    setUserInput(''); // Clear user input
                  }}>
                    Reset
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;