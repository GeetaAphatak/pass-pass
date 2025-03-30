import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import rehypeRaw from 'rehype-raw';
import { assets } from './assets/assets';
import "./Home.css";
import 'ag-grid-community/styles/ag-theme-alpine.css';
// import projectInsights from './assets/project_icons/PI.webp';

const initialMessages = [
  { type: 'bot', text: "Welcome! Tell me what you're interested in, and I'll guide you to the right application." },
];

const Home = () => {
  const [isSidebarVisible, setSidebarVisible] = useState(true);
  const [messages, setMessages] = useState(initialMessages);
  const [userInput, setUserInput] = useState('');
  const [cards, setCards] = useState([]);
  const [history, setHistory] = useState([]);
  const chatContainerRef = useRef(null);

  const toggleSidebar = () => {
    setSidebarVisible(!isSidebarVisible);
  };

  useEffect(() => {
    fetch('http://localhost:8889/config')
      .then(response => response.json())
      .then(data => setCards(data))
      .catch(error => console.error('Error fetching data:', error));
  }, []);

  const handleSendMessage = async () => {
    if (userInput.trim()) {
      const userMessage = { type: 'user', text: userInput };
      setMessages((prevMessages) => [...prevMessages, userMessage]);
      setHistory((prevHistory) => [...prevHistory, { "role": "user", "content": userInput }]);
      setUserInput('');
      try {
        const response = await axios.post("http://localhost:8889/find_app", {
          user_query: userInput,
          history: history
        });

        const { result } = response.data;
        if (result) {
          const convertedText = convertMarkdownLinksToHtml(result);
          const botMessage = { type: 'bot', text: convertedText };

          setMessages((prevMessages) => [...prevMessages, botMessage]);
          setHistory((prevHistory) => [...prevHistory, { "role": "assistant", "content": result }]);
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
    }
  };

  const convertMarkdownLinksToHtml = (text) => {
    const markdownLinkPattern = /\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g;    
    return text.replace(markdownLinkPattern, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  };
  

  useEffect(() => {
    if (chatContainerRef.current) {
      const lastMessage = chatContainerRef.current.lastElementChild;
      if (lastMessage) {
        lastMessage.scrollIntoView({ behavior: 'smooth', block: 'end' });
      }
    }
  }, [messages]);

  return (
    <div className="container">
      <header className="header">
        <div className="header-left">
          {/* <button className="sidebar-toggle-header" type='button' onClick={toggleSidebar} aria-label="Toggle sidebar">
            <span className="hamburger-icon">☰</span>
          </button> */}
          <img src={assets.broadridge_logo} alt="Broadridge Logo" className="logo" />
        </div>
        {/* <img src={assets.broadridge_logo} alt="Broadridge Logo" className="logo" /> */}
        <center><h1 className="heading">Onboarding AI</h1></center>
        <div className="header-icons">
          <a href="http://localhost:3000/" target="_blank" rel="noopener noreferrer" className="icon-link">
            <img src={assets.home_icon} alt="Home" className="home-icon" />
          </a>
          <a href="http://localhost:3000/" target="_blank" rel="noopener noreferrer" className="icon-link">
            <img src={assets.question_icon} alt="Help" className="header-icon" />
          </a>
          <a href="http://localhost:3000/" target="_blank" rel="noopener noreferrer" className="icon-link">
            <img src={assets.user_outline} alt="User" className="header-icon" />
          </a>
        </div>
      </header>

      <div className="main-content">
      <div className={`sidebar ${isSidebarVisible ? 'visible' : 'hidden'}`} id="sidebar">
        <div className="sidebar-content">
          {cards.map(card => (
            <div key={card.name} className="card">
              <a href={card.link} className="card-link" target="_blank" rel="noopener noreferrer" title={card.name}>
                <div className="card-title">
                  <div className="card-img-container">
                    <img src={assets[card.img]} alt={card.name} className="card-img" />
                  </div>
                  <div className="card-name">{card.name}</div>
                </div>
                <div className="card-desc">{card.short_description}</div>
              </a>
            </div>
          ))}
        </div>
      </div>
        <button 
          className={`sidebar-toggle-arrow ${isSidebarVisible ? 'visible' : 'hidden'}`} 
          onClick={toggleSidebar}
          aria-label="Toggle sidebar"
        >
          {isSidebarVisible ? <img src={assets.hide} alt="Hide" className="toggle-img" /> : <img src={assets.unhide} alt="Unhide" className="toggle-img" />}
          
        </button>

        <div className={`content-wrapper ${!isSidebarVisible ? 'expanded' : ''}`}>
          <div className="chat-container">
            <div className="chat-window" ref={chatContainerRef}>
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
                        <img src={assets.bot} alt="Robo" className="bot-img" />
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
                  onChange={(e) => setUserInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
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
                  setMessages(initialMessages);
                  setHistory([]);
                  setUserInput('');
                }} type="button">
                  Reset
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;