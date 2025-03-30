
   import React, { useState } from 'react';

   function App() {
     const [number, setNumber] = useState('');
     const [result, setResult] = useState(null);

     const handleInputChange = (e) => {
       setNumber(e.target.value);
     };

     const handleSubmit = async () => {
       try {
         const response = await fetch('http://127.0.0.1:8889/', {
           method: 'POST',
           headers: {
             'Content-Type': 'application/json',
           },
           body: JSON.stringify({ number: parseInt(number) }),
         });

         const data = await response.json();
         setResult(data.result);
       } catch (error) {
         console.error('Error:', error);
       }
     };

     return (
       <div className="App">
         <h1>Compute Square</h1>
         <input
           type="number"
           value={number}
           onChange={handleInputChange}
           placeholder="Enter a number"
         />
         <button onClick={handleSubmit}>Submit</button>
         {result !== null && <p>Result: {result}</p>}
       </div>
     );
   }

   export default App;



// import logo from './logo.svg';
// import './App.css';
// import Hello from './Hello';  // Import Hello component

// function App() {
//   return (
//     <div className="App">
//       <header className="App-header">
//         <img src={logo} className="App-logo" alt="logo" />
//         <h1>Welcome to My Sample React App</h1>
//            <p>Let's build something amazing!</p>
//            <Hello />  {/* Use Hello component */}
//       </header>
//     </div>
//   );
// }

// export default App;
