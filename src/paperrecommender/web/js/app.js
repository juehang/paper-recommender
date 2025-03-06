// Paper Recommender UI JavaScript

// Global state
const state = {
  currentView: 'home',
  onboardingPapers: [],
  recommendedPapers: [],
  ratings: {},
  config: {},
  isLoading: false,
  progressValue: 0,
  progressTotal: 0,
  progressDescription: '',
};

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  initializeApp();
});

// Initialize the application
async function initializeApp() {
  // Set up navigation
  setupNavigation();
  
  // Load configuration
  await loadConfig();
  
  // Set up event listeners
  setupEventListeners();
  
  // Show the home view by default
  navigateTo('home');
  
  // Check if this is the first startup
  const isFirstStartup = await eel.is_first_startup()();
  if (isFirstStartup) {
    showAlert('Welcome to Paper Recommender! Please complete the onboarding process to get started.', 'info');
    navigateTo('onboarding');
  }
}

// Set up navigation
function setupNavigation() {
  const navLinks = document.querySelectorAll('nav a');
  navLinks.forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const view = e.target.getAttribute('data-view');
      navigateTo(view);
    });
  });
}

// Navigate to a specific view
function navigateTo(view) {
  // Hide all views
  const views = document.querySelectorAll('.view');
  views.forEach(v => v.classList.add('hidden'));
  
  // Show the selected view
  const selectedView = document.getElementById(`${view}-view`);
  if (selectedView) {
    selectedView.classList.remove('hidden');
    
    // Update navigation active state
    const navLinks = document.querySelectorAll('nav a');
    navLinks.forEach(link => {
      link.classList.remove('active');
      if (link.getAttribute('data-view') === view) {
        link.classList.add('active');
      }
    });
    
    // Update current view in state
    state.currentView = view;
    
    // Perform view-specific initialization
    initializeView(view);
  }
}

// Initialize a specific view
async function initializeView(view) {
  switch (view) {
    case 'home':
      // Nothing special needed for home view
      break;
    case 'onboarding':
      await initializeOnboardingView();
      break;
    case 'recommendations':
      await initializeRecommendationsView();
      break;
    case 'settings':
      await initializeSettingsView();
      break;
  }
}

// Load configuration
async function loadConfig() {
  console.log("Starting to load configuration...");
  showLoading(true, 'Loading configuration...');
  try {
    console.log("Calling eel.get_config()...");
    state.config = await eel.get_config()();
    console.log(`Configuration loaded: ${Object.keys(state.config).length} keys`);
    updateConfigDisplay();
  } catch (error) {
    console.error("Error loading configuration:", error);
    showAlert(`Error loading configuration: ${error}`, 'danger');
  } finally {
    showLoading(false);
  }
}

// Update configuration display
function updateConfigDisplay() {
  const configForm = document.getElementById('config-form');
  if (configForm && state.config) {
    // Populate form fields with current config values
    for (const [key, value] of Object.entries(state.config)) {
      const input = configForm.querySelector(`[name="${key}"]`);
      if (input) {
        input.value = value;
      }
    }
  }
}

// Set up event listeners
function setupEventListeners() {
  // Settings form submission
  const configForm = document.getElementById('config-form');
  if (configForm) {
    configForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      await saveConfig();
    });
  }
  
  // Onboarding form submission
  const onboardingForm = document.getElementById('onboarding-form');
  if (onboardingForm) {
    onboardingForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      await submitOnboarding();
    });
  }
  
  // Custom paper form submission
  const customPaperForm = document.getElementById('custom-paper-form');
  if (customPaperForm) {
    customPaperForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      await addCustomPaper();
    });
  }
  
  // Recommendation rating form submission
  const recommendationForm = document.getElementById('recommendation-form');
  if (recommendationForm) {
    recommendationForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      await submitRecommendationRatings();
    });
  }
  
  // Register for progress updates
  eel.expose(updateProgress);
}

// Initialize onboarding view
async function initializeOnboardingView() {
  showLoading(true, 'Preparing onboarding papers...');
  try {
    // Get onboarding papers
    const strategyPapers = await eel.prepare_onboarding_candidates()();
    
    // Flatten the array of strategy papers
    state.onboardingPapers = [];
    let globalIndex = 0;
    
    strategyPapers.forEach((papers, strategyIndex) => {
      const strategyName = ['Random Selection', 'Diverse Selection'][strategyIndex];
      
      papers.forEach(paper => {
        state.onboardingPapers.push({
          ...paper,
          globalIndex,
          strategyName
        });
        globalIndex++;
      });
    });
    
    // Render onboarding papers
    renderOnboardingPapers();
  } catch (error) {
    showAlert(`Error preparing onboarding papers: ${error}`, 'danger');
  } finally {
    showLoading(false);
  }
}

// Render onboarding papers
function renderOnboardingPapers() {
  const container = document.getElementById('onboarding-papers');
  if (!container) return;
  
  // Clear container
  container.innerHTML = '';
  
  if (state.onboardingPapers.length === 0) {
    container.innerHTML = '<div class="alert alert-info">No papers available for onboarding. Try adjusting the time period in settings.</div>';
    return;
  }
  
  // Group papers by strategy
  const papersByStrategy = {};
  state.onboardingPapers.forEach(paper => {
    if (!papersByStrategy[paper.strategyName]) {
      papersByStrategy[paper.strategyName] = [];
    }
    papersByStrategy[paper.strategyName].push(paper);
  });
  
  // Render papers by strategy
  for (const [strategyName, papers] of Object.entries(papersByStrategy)) {
    // Create strategy section
    const strategySection = document.createElement('div');
    strategySection.className = 'strategy-section mb-4';
    
    // Create strategy header
    const strategyHeader = document.createElement('h3');
    strategyHeader.textContent = strategyName;
    strategyHeader.className = 'mb-3';
    strategySection.appendChild(strategyHeader);
    
    // Create papers container
    const papersContainer = document.createElement('div');
    papersContainer.className = 'papers-container';
    
    // Add papers
    papers.forEach(paper => {
      const paperCard = createPaperCard(paper, true);
      papersContainer.appendChild(paperCard);
    });
    
    strategySection.appendChild(papersContainer);
    container.appendChild(strategySection);
  }
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
  
  // Abstract toggle
  const abstractToggle = document.createElement('button');
  abstractToggle.className = 'btn btn-secondary mt-2 mb-2';
  abstractToggle.textContent = 'Show Abstract';
  abstractToggle.addEventListener('click', () => {
    const abstractElement = paperCard.querySelector('.paper-abstract');
    if (abstractElement.classList.contains('hidden')) {
      abstractElement.classList.remove('hidden');
      abstractToggle.textContent = 'Hide Abstract';
    } else {
      abstractElement.classList.add('hidden');
      abstractToggle.textContent = 'Show Abstract';
    }
  });
  paperCard.appendChild(abstractToggle);
  
  // Paper abstract (hidden by default)
  const abstract = document.createElement('div');
  abstract.className = 'paper-abstract hidden';
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
  
  // Create 5 stars
  for (let i = 5; i >= 1; i--) {
    const input = document.createElement('input');
    input.type = 'radio';
    input.name = `rating-${paper.globalIndex}`;
    input.value = i;
    input.id = `rating-${paper.globalIndex}-${i}`;
    
    // Check if this paper already has a rating
    if (state.ratings[paper.globalIndex] === i) {
      input.checked = true;
    }
    
    // Add event listener to update state when rating changes
    input.addEventListener('change', () => {
      if (isOnboarding) {
        state.ratings[paper.globalIndex] = i;
      } else {
        // For recommendations, store ratings differently
        if (!state.recommendationRatings) {
          state.recommendationRatings = {};
        }
        state.recommendationRatings[paper.globalIndex] = i;
      }
    });
    
    const label = document.createElement('label');
    label.setAttribute('for', `rating-${paper.globalIndex}-${i}`);
    label.title = `${i} stars`;
    
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

// Submit onboarding ratings
async function submitOnboarding() {
  // Check if any papers have been rated
  const ratedCount = Object.keys(state.ratings).length;
  if (ratedCount === 0) {
    showAlert('Please rate at least one paper before submitting.', 'warning');
    return;
  }
  
  showLoading(true, 'Submitting ratings...');
  try {
    // Convert ratings to the format expected by the backend
    const ratingsArray = Object.entries(state.ratings).map(([index, rating]) => [parseInt(index), rating]);
    
    // Submit ratings
    const result = await eel.commit_onboarding(ratingsArray)();
    
    // Show success message
    showAlert(`Successfully submitted ${ratedCount} paper ratings. ${result.total_count} papers were added to the database.`, 'success');
    
    // Clear ratings
    state.ratings = {};
    
    // Navigate to recommendations
    navigateTo('recommendations');
  } catch (error) {
    showAlert(`Error submitting ratings: ${error}`, 'danger');
  } finally {
    showLoading(false);
  }
}

// Add a custom paper
async function addCustomPaper() {
  const titleInput = document.getElementById('custom-paper-title');
  const abstractInput = document.getElementById('custom-paper-abstract');
  const linkInput = document.getElementById('custom-paper-link');
  const ratingInput = document.getElementById('custom-paper-rating');
  
  const title = titleInput.value.trim();
  const abstract = abstractInput.value.trim();
  const link = linkInput.value.trim();
  const rating = parseInt(ratingInput.value);
  
  if (!title || !abstract) {
    showAlert('Title and abstract are required.', 'warning');
    return;
  }
  
  if (isNaN(rating) || rating < 1 || rating > 5) {
    showAlert('Please provide a valid rating between 1 and 5.', 'warning');
    return;
  }
  
  showLoading(true, 'Adding custom paper...');
  try {
    const success = await eel.add_custom_paper(title, abstract, link, rating)();
    
    if (success) {
      showAlert('Custom paper added successfully.', 'success');
      
      // Clear form
      titleInput.value = '';
      abstractInput.value = '';
      linkInput.value = '';
      ratingInput.value = '3';
      
      // Ask if user wants to bootstrap the model
      if (confirm('Would you like to update the recommendation model with the new data?')) {
        await bootstrapRecommender();
      }
    } else {
      showAlert('Failed to add custom paper. It may already exist in the database.', 'warning');
    }
  } catch (error) {
    showAlert(`Error adding custom paper: ${error}`, 'danger');
  } finally {
    showLoading(false);
  }
}

// Bootstrap the recommender
async function bootstrapRecommender() {
  showLoading(true, 'Updating recommendation model...');
  try {
    await eel.bootstrap_recommender()();
    showAlert('Recommendation model updated successfully.', 'success');
  } catch (error) {
    showAlert(`Error updating recommendation model: ${error}`, 'danger');
  } finally {
    showLoading(false);
  }
}

// Initialize recommendations view
async function initializeRecommendationsView() {
  showLoading(true, 'Generating recommendations...');
  try {
    // Get recommendations
    const recommendations = await eel.get_recommendations(
      state.config.num_recommendations,
      state.config.exploration_weight,
      state.config.n_nearest_embeddings,
      state.config.gp_num_samples
    )();
    
    state.recommendedPapers = recommendations;
    
    // Add global indices to recommendations
    state.recommendedPapers.forEach((paper, index) => {
      paper.globalIndex = index;
    });
    
    // Render recommendations
    renderRecommendations();
  } catch (error) {
    showAlert(`Error generating recommendations: ${error}`, 'danger');
  } finally {
    showLoading(false);
  }
}

// Render recommendations
function renderRecommendations() {
  const container = document.getElementById('recommendation-papers');
  if (!container) return;
  
  // Clear container
  container.innerHTML = '';
  
  if (state.recommendedPapers.length === 0) {
    container.innerHTML = '<div class="alert alert-info">No recommendations available. Please complete the onboarding process first or adjust the time period in settings.</div>';
    return;
  }
  
  // Create recommendations container
  const recommendationsContainer = document.createElement('div');
  recommendationsContainer.className = 'recommendations-container';
  
  // Add papers
  state.recommendedPapers.forEach(paper => {
    const paperCard = createPaperCard(paper, false);
    recommendationsContainer.appendChild(paperCard);
  });
  
  container.appendChild(recommendationsContainer);
  
  // Add submit button
  const submitButton = document.createElement('button');
  submitButton.className = 'btn btn-primary mt-3';
  submitButton.textContent = 'Submit Ratings';
  submitButton.addEventListener('click', submitRecommendationRatings);
  container.appendChild(submitButton);
}

// Submit recommendation ratings
async function submitRecommendationRatings() {
  // Check if any papers have been rated
  if (!state.recommendationRatings || Object.keys(state.recommendationRatings).length === 0) {
    showAlert('Please rate at least one paper before submitting.', 'warning');
    return;
  }
  
  showLoading(true, 'Submitting ratings...');
  try {
    // Convert ratings to the format expected by the backend
    const ratingsArray = [];
    
    for (const [index, rating] of Object.entries(state.recommendationRatings)) {
      const paperIndex = parseInt(index);
      const paper = state.recommendedPapers[paperIndex];
      
      ratingsArray.push({
        document: paper.document,
        link: paper.link,
        rating: rating
      });
    }
    
    // Submit ratings
    const result = await eel.add_recommendation_ratings(ratingsArray)();
    
    // Show success message
    showAlert(`Successfully submitted ${ratingsArray.length} paper ratings.`, 'success');
    
    // Ask if user wants to bootstrap the model
    if (confirm('Would you like to update the recommendation model with the new ratings?')) {
      await bootstrapRecommender();
      
      // Refresh recommendations
      await initializeRecommendationsView();
    }
    
    // Clear ratings
    state.recommendationRatings = {};
  } catch (error) {
    showAlert(`Error submitting ratings: ${error}`, 'danger');
  } finally {
    showLoading(false);
  }
}

// Initialize settings view
async function initializeSettingsView() {
  // Load config if not already loaded
  if (!state.config || Object.keys(state.config).length === 0) {
    await loadConfig();
  }
}

// Save configuration
async function saveConfig() {
  const configForm = document.getElementById('config-form');
  if (!configForm) return;
  
  // Get form data
  const formData = new FormData(configForm);
  const newConfig = {};
  
  // Convert form data to config object
  for (const [key, value] of formData.entries()) {
    // Convert numeric values
    if (!isNaN(value) && value !== '') {
      if (value.includes('.')) {
        newConfig[key] = parseFloat(value);
      } else {
        newConfig[key] = parseInt(value);
      }
    } else {
      newConfig[key] = value;
    }
  }
  
  showLoading(true, 'Saving configuration...');
  try {
    await eel.save_config(newConfig)();
    state.config = newConfig;
    showAlert('Configuration saved successfully.', 'success');
  } catch (error) {
    showAlert(`Error saving configuration: ${error}`, 'danger');
  } finally {
    showLoading(false);
  }
}

// Show loading indicator
function showLoading(isLoading, description = '') {
  state.isLoading = isLoading;
  
  const loadingOverlay = document.getElementById('loading-overlay');
  const loadingText = document.getElementById('loading-text');
  
  if (loadingOverlay && loadingText) {
    if (isLoading) {
      loadingText.textContent = description;
      loadingOverlay.classList.remove('hidden');
    } else {
      loadingOverlay.classList.add('hidden');
    }
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
  state.progressValue = current;
  state.progressTotal = total;
  state.progressDescription = description;
  
  const progressBar = document.getElementById('progress-bar');
  const progressText = document.getElementById('progress-text');
  
  if (progressBar && progressText) {
    const percentage = total > 0 ? (current / total) * 100 : 0;
    progressBar.style.width = `${percentage}%`;
    progressText.textContent = `${description}: ${current}/${total} (${percentage.toFixed(0)}%)`;
    
    if (percentage === 100) {
      setTimeout(() => {
        progressBar.style.width = '0%';
        progressText.textContent = '';
      }, 1000);
    }
  }
}

// Expose functions to Python
eel.expose(showAlert);
eel.expose(showLoading);
eel.expose(updateProgress);
