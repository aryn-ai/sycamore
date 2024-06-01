import React from "react";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import PdfViewer from "./pages/PDFViewer/PdfViewer";
import HomePage from "./pages/HomePage";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="viewPdf" element={<PdfViewer />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
