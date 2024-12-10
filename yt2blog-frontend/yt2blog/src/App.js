import React, { useState } from "react";
import "./App.css";

function App() {
  const [youtubeLink, setYoutubeLink] = useState("");
  const [email, setEmail] = useState("");
  const [loading, setLoading] = useState(false);
  const [confirmation, setConfirmation] = useState(false);

  const handleInputChange = (e) => {
    setYoutubeLink(e.target.value);
  };

  const handleEmailChange = (e) => {
    setEmail(e.target.value);
  };

  const handleGenerateBlog = async () => {
    if (!youtubeLink || !email) {
      alert("Please enter both a YouTube video link and your email address!");
      return;
    }

    setLoading(true);
    try {
      const res = await fetch(
        `http://localhost:8000/blogs/?url=${encodeURIComponent(
          youtubeLink
        )}&email=${encodeURIComponent(email)}`,
        {
          method: "GET",
        }
      );

      if (!res.ok) {
        throw new Error("Failed to process your request");
      }

      setConfirmation(true);
    } catch (error) {
      alert("An error occurred: " + error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setYoutubeLink("");
    setEmail("");
    setConfirmation(false);
  };

  return (
    <div className="app">
      {loading && (
        <div className="loading-screen">
          <div className="spinner"></div>
          <p>Processing your request... Please wait.</p>
        </div>
      )}
      <header className="header">
        <img src="logo.png" alt="Vid2Blog Logo" className="logo"/>
      </header>

      <div className="container">
        {!confirmation ? (
          <div className="input-container">
            <input
              type="text"
              placeholder="Paste YouTube video link..."
              value={youtubeLink}
              onChange={handleInputChange}
              className="input-field"
            />
            <input
              type="email"
              placeholder="Enter your email address..."
              value={email}
              onChange={handleEmailChange}
              className="input-field"
            />
            <button onClick={handleGenerateBlog} className="generate-button">
              Generate Blog
            </button>
          </div>
        ) : (
          <div className="confirmation-container">
            <p>Email will be sent to {email}</p>
            <button onClick={handleReset} className="generate-button">
              Generate Another Blog
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
