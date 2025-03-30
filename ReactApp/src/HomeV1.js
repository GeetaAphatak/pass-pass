import React, { useState, useEffect, useRef } from 'react';
import './App.css';  // Import the CSS file
import ReactMarkdown from 'react-markdown';
import rehypeRaw from 'rehype-raw';
import Home from './Home';

const OnboardingAI = () => {
  const [cards, setCards] = useState([]);
  const [userInput, setUserInput] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const chatContainerRef = useRef(null);

  useEffect(() => {
    // Load cards initially
    fetch('http://localhost:8889/config')
      .then(response => response.json())
      .then(data => setCards(data))
      .catch(error => console.error('Error fetching data:', error));
  }, []);

  const submitAiQuery = () => {
    if (!userInput.trim()) return;

    // Add user message to chat history once
    const userMessage = { role: 'user', content: userInput };
    setChatHistory(current => [...current, userMessage]);

    // Clear the user input
    setUserInput('');

    fetch('http://localhost:8889/find_app', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_query: userInput, history: [...chatHistory, userMessage] }),
    })
      .then(response => response.json())
      .then(result => {
        if (result && result.result) {
          const aiResponse = convertMarkdownLinksToHtml(result.result);
          setChatHistory(current => [...current, { role: 'ai', content: aiResponse }]);
        } else {
          setChatHistory(current => [...current, { role: 'ai', content: "An unexpected error occurred." }]);
        }
      })
      .catch(() => {
        setChatHistory(current => [...current, { role: 'ai', content: "Error occurred during processing." }]);
      });
  };

  const convertMarkdownLinksToHtml = text => {
    const markdownLinkPattern = /\[([^\]]+)\]\((https?:\/\/[^\)]+)\)/g;
    return text.replace(markdownLinkPattern, '<a href="$2" target="_blank">$1</a>');
  };

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [chatHistory]);

  return (
    <div>
      <div className="header">
        <div className="Broadridge-container">
          <img src="./br-primary-blue-logo.svg" alt="Broadridge" className="logo" />
        </div>
      </div>
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
        <div className="main-content">
          <div className="section-container ai-query-section">
            <h2>Describe what you're looking for:</h2>
            <div className="input-wrapper">
              <textarea
                id="aiQueryInput"
                className="ai-query-input"
                placeholder="Type your query here..."
                wrap="soft"
                value={userInput}
                onChange={(e) => setUserInput(e.target.value)}
              />
              <button onClick={submitAiQuery} className="ai-query-button">Submit</button>
            </div>
            <div className="chat-container" ref={chatContainerRef}>
              {chatHistory.map((message, index) => (
                <div key={index} className={`chat-bubble ${message.role === 'user' ? 'user-bubble' : 'ai-bubble'}`}>
                  {message.role === 'ai' ? 
                    // <ReactMarkdown>{message.content}</ReactMarkdown> : 
                    // message.content}
                   <ReactMarkdown rehypePlugins={[rehypeRaw]}>{message.content}</ReactMarkdown>: 
                   message.content}

              </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;