import React from 'react';
import { assets } from '../assets/assets';
import './ChatInput.css';

const ChatInput = ({ userInput, setUserInput, handleSendMessage, onReset, isSidebarVisible }) => {
  return (
    <div className={`footer ${!isSidebarVisible ? 'expanded' : ''}`} style={{ left: isSidebarVisible ? 'var(--sidebar-width)' : 'var(--toggle-width)' }}>
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
        <button onClick={onReset} type="button">
          Reset
        </button>
      </div>
    </div>
  );
};

export default ChatInput;
