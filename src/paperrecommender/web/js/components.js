// Paper Recommender UI Components
// This file contains UI components and utility functions

// Create namespace if it doesn't exist
window.PaperRecommender = window.PaperRecommender || {};

// UI Components module
(function(namespace) {
    // Progress Tracker Class - For UI Modernization
    class ProgressTracker {
        constructor() {
            this.footer = document.getElementById('progress-footer');
            this.icon = document.getElementById('progress-icon');
            this.text = document.getElementById('progress-text');
            this.progressBar = document.getElementById('progress-bar');
            this._timeoutId = null;
            this._activeOperations = new Set();
        }

        /**
         * Show progress indicator with message and percentage
         * @param {string} message - Progress message to display
         * @param {number} percentage - Progress percentage (0-100)
         * @param {string} state - State ('loading', 'complete', 'error')
         * @param {string} operationId - Optional unique ID for operation
         * @param {number} timeout - Optional auto-hide timeout in ms
         */
        show(message, percentage = 0, state = 'loading', operationId = null, timeout = 0) {
            this.footer.classList.remove('hidden');
            this.text.textContent = message;
            this.progressBar.style.width = `${percentage}%`;
            this.updateIcon(state);
            
            // Track operation if ID provided
            if (operationId) {
                this._activeOperations.add(operationId);
            }
            
            // Set auto-hide timeout if provided
            if (timeout > 0) {
                if (this._timeoutId) {
                    clearTimeout(this._timeoutId);
                }
                
                this._timeoutId = setTimeout(() => {
                    this.hide(operationId);
                }, timeout);
            }
        }

        /**
         * Hide progress indicator
         * @param {string} operationId - Optional operation ID to complete
         */
        hide(operationId = null) {
            // If operation ID provided, remove from active operations
            if (operationId) {
                this._activeOperations.delete(operationId);
                
                // Don't hide if other operations are still active
                if (this._activeOperations.size > 0) {
                    return;
                }
            }
            
            // Clear any pending timeout
            if (this._timeoutId) {
                clearTimeout(this._timeoutId);
                this._timeoutId = null;
            }
            
            this.footer.classList.add('hidden');
        }

        /**
         * Update progress icon based on state
         * @param {string} state - The current state
         */
        updateIcon(state) {
            // Use SVG icons for better styling and consistency
            const icons = {
                'loading': `
                    <svg class="animate-spin w-5 h-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                `,
                'complete': `
                    <svg class="w-5 h-5 text-green-500" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                        <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd" />
                    </svg>
                `,
                'error': `
                    <svg class="w-5 h-5 text-red-500" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                        <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clip-rule="evenodd" />
                    </svg>
                `
            };
            
            this.icon.innerHTML = icons[state] || icons['loading'];
        }

        /**
         * Get singleton instance of ProgressTracker
         * @returns {ProgressTracker} The singleton instance
         */
        static getInstance() {
            if (!this.instance) {
                this.instance = new ProgressTracker();
            }
            return this.instance;
        }
    }

    // Show loading indicator
    function showLoading(isLoading, description = '') {
        // Use the ProgressTracker for loading indication
        const progress = ProgressTracker.getInstance();
        
        if (isLoading) {
            progress.show(description, 0, 'loading', 'global-loading');
        } else {
            progress.hide('global-loading');
        }
    }

    // Show alert message
    function showAlert(message, type = 'info') {
        const alertsContainer = document.getElementById('alerts-container');
        if (!alertsContainer) return;
        
        // Create alert element
        const alert = document.createElement('div');
        alert.className = `alert alert-${type}`;
        alert.textContent = message;
        
        // Add close button
        const closeButton = document.createElement('button');
        closeButton.className = 'close';
        closeButton.innerHTML = '&times;';
        closeButton.addEventListener('click', () => {
            alert.remove();
        });
        
        alert.appendChild(closeButton);
        alertsContainer.appendChild(alert);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            alert.remove();
        }, 5000);
    }

    // Update progress bar
    function updateProgress(current, total, description) {
        // Use the ProgressTracker for progress indication
        const progress = ProgressTracker.getInstance();
        
        if (total > 0) {
            const percentage = (current / total) * 100;
            progress.show(`${description}: ${current}/${total}`, percentage, 'loading', 'progress-update');
            
            if (percentage === 100) {
                setTimeout(() => {
                    progress.show(`${description} complete`, 100, 'complete', 'progress-update', 2000);
                }, 500);
            }
        }
    }

    // Function to update star rating visual state
    function updateStarRating(container, selectedValue) {
        // Get all star labels in this container
        const labels = container.querySelectorAll('label');
        
        // Update each label based on its value compared to the selected value
        labels.forEach(label => {
            const starValue = parseInt(label.dataset.value);
            
            // If star value is less than or equal to selected value, highlight it
            if (starValue <= selectedValue) {
                label.style.backgroundImage = 'url(\'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M12 17.27L18.18 21l-1.64-7.03L22 9.24l-7.19-.61L12 2 9.19 8.63 2 9.24l5.46 4.73L5.82 21z" fill="%23f39c12"/></svg>\')';
            } else {
                label.style.backgroundImage = 'url(\'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M12 17.27L18.18 21l-1.64-7.03L22 9.24l-7.19-.61L12 2 9.19 8.63 2 9.24l5.46 4.73L5.82 21z" fill="%23ddd"/></svg>\')';
            }
        });
    }

    // Create a paper card element
    function createPaperCard(paper, isOnboarding = false) {
        const paperCard = document.createElement('div');
        paperCard.className = 'paper-card';
        paperCard.dataset.index = paper.globalIndex;
        
        // Paper title
        const title = document.createElement('h3');
        title.className = 'paper-title';
        title.textContent = paper.title;
        paperCard.appendChild(title);
        
        // Paper link
        if (paper.link) {
            const link = document.createElement('a');
            link.className = 'paper-link';
            link.href = paper.link;
            link.target = '_blank';
            link.textContent = 'View Paper';
            paperCard.appendChild(link);
        }
        
        // Paper abstract (visible by default)
        const abstract = document.createElement('div');
        abstract.className = 'paper-abstract';
        abstract.textContent = paper.abstract;
        paperCard.appendChild(abstract);
        
        // Rating section
        const ratingSection = document.createElement('div');
        ratingSection.className = 'paper-rating';
        
        // Rating label
        const ratingLabel = document.createElement('label');
        ratingLabel.textContent = 'Rate this paper:';
        ratingSection.appendChild(ratingLabel);
        
        // Star rating
        const ratingContainer = document.createElement('div');
        ratingContainer.className = 'rating';
        
        // Create 5 stars (left to right)
        for (let i = 1; i <= 5; i++) {
            const input = document.createElement('input');
            input.type = 'radio';
            input.name = `rating-${paper.globalIndex}`;
            input.value = i;
            input.id = `rating-${paper.globalIndex}-${i}`;
            
            // Check if this paper already has a rating
            if (window.PaperRecommender.core.state.ratings[paper.globalIndex] === i) {
                input.checked = true;
            }
            
            // Add event listener to update state when rating changes
            input.addEventListener('change', () => {
                if (isOnboarding) {
                    window.PaperRecommender.core.state.ratings[paper.globalIndex] = i;
                } else {
                    // For recommendations, store ratings differently
                    if (!window.PaperRecommender.core.state.recommendationRatings) {
                        window.PaperRecommender.core.state.recommendationRatings = {};
                    }
                    window.PaperRecommender.core.state.recommendationRatings[paper.globalIndex] = i;
                }
                
                // Update the visual state of the stars
                updateStarRating(ratingContainer, i);
            });
            
            const label = document.createElement('label');
            label.setAttribute('for', `rating-${paper.globalIndex}-${i}`);
            label.title = `${i} stars`;
            label.dataset.value = i; // Add data attribute for CSS selection
            
            ratingContainer.appendChild(input);
            ratingContainer.appendChild(label);
        }
        
        ratingSection.appendChild(ratingContainer);
        paperCard.appendChild(ratingSection);
        
        // If this is a recommendation, show prediction info
        if (!isOnboarding && paper.predicted_rating) {
            const predictionInfo = document.createElement('div');
            predictionInfo.className = 'prediction-info mt-2';
            
            const predictionText = document.createElement('p');
            predictionText.innerHTML = `<strong>Predicted Rating:</strong> ${paper.predicted_rating.toFixed(2)}/5.0`;
            predictionInfo.appendChild(predictionText);
            
            if (paper.lower_bound && paper.upper_bound) {
                const rangeText = document.createElement('p');
                rangeText.innerHTML = `<strong>Rating Range:</strong> [${paper.lower_bound.toFixed(2)}, ${paper.upper_bound.toFixed(2)}]`;
                predictionInfo.appendChild(rangeText);
            }
            
            paperCard.appendChild(predictionInfo);
        }
        
        return paperCard;
    }

    // Export public functions and classes
    namespace.components = {
        ProgressTracker,
        showLoading,
        showAlert,
        updateProgress,
        updateStarRating,
        createPaperCard
    };
})(window.PaperRecommender);

// Define functions for Eel to call
function showProgress(message, percentage, state, operationId, timeout) {
    window.PaperRecommender.components.ProgressTracker.getInstance().show(message, percentage, state, operationId, timeout);
}

function hideProgress(operationId) {
    window.PaperRecommender.components.ProgressTracker.getInstance().hide(operationId);
}

function updateProgress(current, total, description) {
    console.log(`Progress update: ${description} - ${current}/${total}`);
    const percentage = (current / total * 100) || 0;
    window.PaperRecommender.components.ProgressTracker.getInstance().show(
        `${description}: ${current}/${total}`,
        percentage,
        'loading',
        'progress-update'
    );
}

// Expose functions to Eel
if (typeof eel !== 'undefined') {
    eel.expose(showProgress, 'show_progress');
    eel.expose(hideProgress, 'hide_progress');
    eel.expose(updateProgress, 'updateProgress');
}