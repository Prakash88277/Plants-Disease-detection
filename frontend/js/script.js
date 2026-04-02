/**
 * PlantCare AI - Main Script
 * Handles UI interactions, mobile menu, drag/drop, validation & API logic.
 */

document.addEventListener('DOMContentLoaded', () => {
    // 1. Navbar Mobile Menu Toggle
    const mobileMenuBtn = document.getElementById('mobile-menu-btn');
    const closeMenuBtn = document.getElementById('close-menu-btn');
    const navLinks = document.getElementById('nav-links');
    const mobileOverlay = document.getElementById('mobile-overlay');
    const navLinksItems = document.querySelectorAll('.nav-links a');

    function toggleMobileMenu() {
        navLinks.classList.toggle('active');
        mobileOverlay.classList.toggle('active');
        document.body.style.overflow = navLinks.classList.contains('active') ? 'hidden' : '';
    }

    function closeMobileMenu() {
        navLinks.classList.remove('active');
        mobileOverlay.classList.remove('active');
        document.body.style.overflow = '';
    }

    if (mobileMenuBtn && closeMenuBtn && mobileOverlay) {
        mobileMenuBtn.addEventListener('click', toggleMobileMenu);
        closeMenuBtn.addEventListener('click', closeMobileMenu);
        mobileOverlay.addEventListener('click', closeMobileMenu);
    }

    // Close mobile menu on link click
    navLinksItems.forEach(link => {
        link.addEventListener('click', closeMobileMenu);
    });

    // Navbar scroll effect
    const navbar = document.getElementById('navbar');
    window.addEventListener('scroll', () => {
        if (window.scrollY > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
    });

    // 2. Elements for Upload functionality
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('file-input');
    const uploadContent = document.getElementById('upload-content');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const removeBtn = document.getElementById('remove-btn');
    const detectBtn = document.getElementById('detect-btn');
    const fileNameDisplay = document.getElementById('file-name-display');
    const uploadBtnTrigger = document.getElementById('upload-btn-trigger');
    
    // Elements for State/Results
    const loadingState = document.getElementById('loading');
    const resultSection = document.getElementById('result-section');
    const resPlant = document.getElementById('res-plant');
    const resDisease = document.getElementById('res-disease');
    const resConfidence = document.getElementById('res-confidence');

    // 3. Setup Drag and Drop functionality
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        if(dropArea) dropArea.addEventListener(eventName, preventDefaults, false);
        window.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        if(dropArea) dropArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        if(dropArea) dropArea.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        dropArea.classList.add('dragover');
    }

    function unhighlight(e) {
        dropArea.classList.remove('dragover');
    }

    if(dropArea) {
        dropArea.addEventListener('drop', handleDrop, false);
    }

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    if(fileInput) {
        fileInput.addEventListener('change', function() {
            handleFiles(this.files);
        });
    }

    // Handle file processing
    function handleFiles(files) {
        if (files.length === 0) return;
        
        const file = files[0];
        
        // Validation
        if (!validateFileType(file)) {
            alert('Please upload a valid image file (JPG, JPEG, or PNG).');
            return;
        }

        if (!validateFileSize(file)) {
            alert('File size exceeds 5MB limit. Please upload a smaller image.');
            return;
        }

        fileNameDisplay.textContent = file.name;
        previewImage(file);
    }

    function validateFileType(file) {
        const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
        return validTypes.includes(file.type);
    }

    function validateFileSize(file) {
        const maxSize = 5 * 1024 * 1024; // 5MB in bytes
        return file.size <= maxSize;
    }

    function previewImage(file) {
        const reader = new FileReader();
        
        reader.readAsDataURL(file);
        
        reader.onloadend = function() {
            imagePreview.src = reader.result;
            
            uploadContent.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            
            if(uploadBtnTrigger) uploadBtnTrigger.style.display = 'inline-flex';
            if(detectBtn) detectBtn.disabled = false;
            if(resultSection) resultSection.classList.add('hidden');
        }
    }

    // 4. Remove Image
    if(removeBtn) {
        removeBtn.addEventListener('click', function(e) {
            e.preventDefault();
            resetUploadState();
        });
    }

    function resetUploadState() {
        if(fileInput) fileInput.value = '';
        
        previewContainer.classList.add('hidden');
        uploadContent.classList.remove('hidden');
        imagePreview.src = '';
        
        if(uploadBtnTrigger) uploadBtnTrigger.style.display = 'none';
        if(detectBtn) detectBtn.disabled = true;
        
        if(resultSection) resultSection.classList.add('hidden');
        if(loadingState) loadingState.classList.add('hidden');
    }

    // 5. Real API Request Integration
    if(detectBtn) {
        detectBtn.addEventListener('click', async function() {
            // Check if file is provided
            if (!fileInput.files || fileInput.files.length === 0) {
                showError("Please upload a valid image before detecting.");
                return;
            }

            const file = fileInput.files[0];
            
            // Clean up any existing errors
            clearError();

            detectBtn.disabled = true;
            loadingState.classList.remove('hidden');
            resultSection.classList.add('hidden');
            
            loadingState.scrollIntoView({ behavior: 'smooth', block: 'center' });
            
            try {
                const formData = new FormData();
                formData.append("file", file);
                formData.append("t", Date.now()); // Defeat caching dynamically
                
                console.log("Sending request...");
                
                const response = await fetch("http://127.0.0.1:8000/predict", {
                    method: "POST",
                    body: formData,
                    cache: "no-store"
                });
                
                if (!response.ok) {
                    throw new Error("Failed to connect to server or invalid response");
                }
                
                const data = await response.json();
                console.log("Response:", data);
                
                loadingState.classList.add('hidden');
                
                // Update UI dynamically
                resPlant.textContent = data.plant;
                resDisease.textContent = data.disease;
                
                // We're expecting data.confidence to already be a formatted string (e.g. '96.7%') 
                // based on our backend implementation. If the user expects manual appendage:
                resConfidence.textContent = data.confidence.includes('%') ? data.confidence : data.confidence + "%";
                
                resultSection.classList.remove('hidden');
                resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            } catch (error) {
                console.error("Error:", error);
                loadingState.classList.add('hidden');
                showError("Failed to connect to server. Please ensure the backend is running.");
            } finally {
                detectBtn.disabled = false;
            }
        });
    }

    // 6. Model Analysis Toggle
    const analysisBtn = document.getElementById('analysis-btn');
    const analysisPanel = document.getElementById('analysis-panel');

    if (analysisBtn && analysisPanel) {
        analysisBtn.addEventListener('click', () => {
            if (analysisPanel.classList.contains('show')) {
                // Hide panel
                analysisPanel.classList.remove('show');
                setTimeout(() => {
                    analysisPanel.classList.add('hidden');
                }, 400); // Wait for transition
            } else {
                // Show panel
                analysisPanel.classList.remove('hidden');
                // Force reflow to ensure transition runs
                void analysisPanel.offsetWidth;
                analysisPanel.classList.add('show');
                // Scroll slightly
                setTimeout(() => {
                    analysisPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }, 100);
            }
        });
    }

    // Helper to show red alert box in UI for errors
    function showError(message) {
        clearError(); // Remove existing if any
        
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-alert';
        errorDiv.id = 'dynamic-error-alert';
        errorDiv.style.backgroundColor = '#fde8e8';
        errorDiv.style.border = '1px solid #f98080';
        errorDiv.style.color = '#c81e1e';
        errorDiv.style.padding = '12px 16px';
        errorDiv.style.borderRadius = '8px';
        errorDiv.style.marginBottom = '20px';
        errorDiv.style.fontWeight = '500';
        errorDiv.style.display = 'flex';
        errorDiv.style.alignItems = 'center';
        errorDiv.style.gap = '10px';
        
        errorDiv.innerHTML = `<i class="fa-solid fa-circle-exclamation"></i> <span>${message}</span>`;
        
        // Insert right before the drop-area inside upload-wrapper
        const uploadWrapper = document.querySelector('.upload-wrapper');
        const dropAreaElement = document.getElementById('drop-area');
        if (uploadWrapper && dropAreaElement) {
            uploadWrapper.insertBefore(errorDiv, dropAreaElement);
        }
    }

    // Helper to clear errors
    function clearError() {
        const existingAlert = document.getElementById('dynamic-error-alert');
        if (existingAlert) {
            existingAlert.remove();
        }
    }

});
