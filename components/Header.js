import React from 'react';
import { assets } from '../assets/assets';
import './Header.css';

const Header = () => {
  return (
    <header className="header">
      <div className="header-left">
        <img src={assets.broadridge_logo} alt="Broadridge Logo" className="logo" />
      </div>
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
  );
};

export default Header;
