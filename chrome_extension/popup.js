// New function to handle file upload
async function uploadFile(file) {
    if (!file) {
        alert("Please select an image.");
        return;
    }
    
    console.log('Starting upload for file:', file.name); // Add logging
    
    // Update the drop zone text to indicate file selection
    document.getElementById("dropZone").innerText = "Uploading...";
    
    let formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("http://127.0.0.1:5000/api/upload", {
            method: "POST",
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('Upload response:', data); // Add logging
        
        if (data.error) {
            document.getElementById("result").innerText = "Error: " + data.error;
            document.getElementById("dropZone").innerText = "Drop image here";
        } else {
            document.getElementById("result").innerText = `Prediction: ${data.prediction} (Confidence: ${data.confidence}%)`;
            document.getElementById("dropZone").innerText = "Drop another image here";
        }
    } catch (error) {
        console.error("Upload error:", error);
        document.getElementById("result").innerText = "Failed to connect to the server.";
        document.getElementById("dropZone").innerText = "Drop image here";
    }
}

// Handle click on Analyze button
document.getElementById("uploadBtn").addEventListener("click", async function() {
    let file = document.getElementById("fileInput").files[0];
    uploadFile(file);
});

// Trigger file input when drop zone is clicked
document.getElementById("dropZone").addEventListener("click", function() {
    document.getElementById("fileInput").click();
});

// Automatically upload when a file is selected from the file input
document.getElementById("fileInput").addEventListener("change", function(event) {
    let file = event.target.files[0];
    if(file) {
        uploadFile(file);
    }
});

// Drag & Drop events for drop zone
let dropZone = document.getElementById("dropZone");

dropZone.addEventListener("dragover", function(event) {
    event.preventDefault();
    event.stopPropagation();
    dropZone.style.borderColor = "green";
});

dropZone.addEventListener("dragleave", function(event) {
    event.preventDefault();
    event.stopPropagation();
    dropZone.style.borderColor = "#ccc";
});

dropZone.addEventListener("drop", async function(event) {
    event.preventDefault();
    event.stopPropagation();
    dropZone.style.borderColor = "#ccc";
    
    // Get the dropped files
    const files = event.dataTransfer.files;
    if (files.length > 0) {
        const file = files[0];
        
        // Create a new FileList-like object
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        
        // Update the file input with the dropped file
        const fileInput = document.getElementById("fileInput");
        fileInput.files = dataTransfer.files;
        
        // Upload the file
        uploadFile(file);
    }
});
