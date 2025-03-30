import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import "./Home.css";
import 'ag-grid-community/styles/ag-theme-alpine.css';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import ChatWindow from './components/ChatWindow';
import ChatInput from './components/ChatInput';

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

  const handleReset = () => {
    setMessages(initialMessages);
    setHistory([]);
    setUserInput('');
  };

  return (
    <div className="container">
      <Header />
      <div className="main-content">
        <Sidebar 
          isVisible={isSidebarVisible} 
          cards={cards} 
          onToggle={toggleSidebar}
        />
        <div className={`content-wrapper ${!isSidebarVisible ? 'expanded' : ''}`}>
          <div className="chat-container">
            <ChatWindow 
              messages={messages} 
              chatContainerRef={chatContainerRef}
            />
            <ChatInput 
              userInput={userInput}
              setUserInput={setUserInput}
              handleSendMessage={handleSendMessage}
              onReset={handleReset}
              isSidebarVisible={isSidebarVisible}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;
