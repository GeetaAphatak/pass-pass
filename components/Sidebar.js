import React from 'react';
import { assets } from '../assets/assets';
import './Sidebar.css';

const Sidebar = ({ isVisible, cards, onToggle }) => {
  return (
    <>
      <div className={`sidebar ${isVisible ? 'visible' : 'hidden'}`} id="sidebar">
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
        className={`sidebar-toggle-arrow ${isVisible ? 'visible' : 'hidden'}`} 
        onClick={onToggle}
        aria-label="Toggle sidebar"
        title={isVisible ? 'Hide sidebar' : 'Show sidebar'}
      >
        <img src={assets.hide} alt={isVisible ? 'Hide' : 'Show'} className="toggle-img" />
      </button>
    </>
  );
};

export default Sidebar;
