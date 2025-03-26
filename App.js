import React, { useState, useEffect } from "react";
import axios from "axios";

function App() {
  const [imageSrc, setImageSrc] = useState("");
  const [predictedChar, setPredictedChar] = useState("?");
  const [savedText, setSavedText] = useState("");

  useEffect(() => {
    const eventSource = new EventSource("http://localhost:5000/video_feed");

    eventSource.onmessage = (event) => {
      const [imageData, character] = event.data.split("|");
      setImageSrc(imageData);
      setPredictedChar(character);
    };

    axios.get("http://localhost:5000/get_text").then((res) => {
      setSavedText(res.data.saved_text);
    });

    return () => eventSource.close();
  }, []);

  const handleKeyPress = (event) => {
    if (event.key === " ") {
      axios.post("http://localhost:5000/save_character", { character: predictedChar })
        .then((res) => setSavedText(res.data.saved_text))
        .catch((err) => console.error("Error saving character:", err));
    }
  };

  useEffect(() => {
    window.addEventListener("keydown", handleKeyPress);
    return () => window.removeEventListener("keydown", handleKeyPress);
  }, [predictedChar]);

  return (
    <div style={{ textAlign: "center", padding: "20px" }}>
      <h1>Hand Gesture Recognition</h1>
      {imageSrc && <img src={imageSrc} alt="Video Stream" style={{ width: "50%", border: "2px solid black" }} />}
      <h2>Predicted Character: {predictedChar}</h2>
      <h2>Saved Text: {savedText}</h2>
      <p>Press <strong>Spacebar</strong> to save the predicted letter</p>
    </div>
  );
}

export default App;
