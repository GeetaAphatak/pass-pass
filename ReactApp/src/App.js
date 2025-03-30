import './App.css';
// import Home from "./HomeV1"; // Version one where the text box is at the top of the page. Colors are also a little different from the Generic one
import Home from "./Home"; //Generic one, getting it from Conversion Methodology UI 

function App() {
  return (
    <div className="App">
      {<Home />}
    </div>
  );
}

export default App;
