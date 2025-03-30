import React from 'react';
import ReactMarkdown from 'react-markdown';
import rehypeRaw from 'rehype-raw';
import { assets } from '../assets/assets';
import './ChatWindow.css';

const ChatWindow = ({ messages, chatContainerRef }) => {
  return (
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
  );
};

export default ChatWindow;
