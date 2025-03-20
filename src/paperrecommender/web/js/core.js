// Paper Recommender UI Core
// This file contains core application functionality

// Extend PaperRecommender namespace
(function(namespace) {
    // Get components from the namespace
    const components = namespace.components || {};
    const { showAlert, showLoading } = components;

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
        recommendationRatings: {},
        databaseDocuments: [],
        semanticSearchResults: null,
        semanticSearchQuery: null
    };

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
        // Add event listeners to navigation links in the nav bar
        const navLinks = document.querySelectorAll('nav a');
        navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const view = e.target.getAttribute('data-view');
                navigateTo(view);
            });
        });
        
        // Add event listeners to all other elements with data-view attribute (like buttons on the home page)
        const otherViewLinks = document.querySelectorAll('a[data-view]:not(nav a)');
        otherViewLinks.forEach(link => {
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
        const views = namespace.views || {};
        
        switch (view) {
            case 'home':
                // Nothing special needed for home view
                break;
            case 'onboarding':
                if (views.initializeOnboardingView) {
                    await views.initializeOnboardingView();
                }
                break;
            case 'recommendations':
                if (views.initializeRecommendationsView) {
                    await views.initializeRecommendationsView();
                }
                break;
            case 'database':
                if (views.initializeDatabaseView) {
                    await views.initializeDatabaseView();
                }
                break;
            case 'visualization':
                if (views.initializeVisualizationView) {
                    await views.initializeVisualizationView();
                }
                break;
            case 'settings':
                if (views.initializeSettingsView) {
                    await views.initializeSettingsView();
                }
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
                if (namespace.views && namespace.views.submitOnboarding) {
                    await namespace.views.submitOnboarding();
                }
            });
        }
        
        // Custom paper form submission
        const customPaperForm = document.getElementById('custom-paper-form');
        if (customPaperForm) {
            customPaperForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                if (namespace.views && namespace.views.addCustomPaper) {
                    await namespace.views.addCustomPaper();
                }
            });
        }
        
        // Recommendation rating form submission
        const recommendationForm = document.getElementById('recommendation-form');
        if (recommendationForm) {
            recommendationForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                if (namespace.views && namespace.views.submitRecommendationRatings) {
                    await namespace.views.submitRecommendationRatings();
                }
            });
        }
        
        // Database time filter select
        const dbTimeFilter = document.getElementById('db-time-filter');
        if (dbTimeFilter) {
            dbTimeFilter.addEventListener('change', () => {
                if (namespace.views && namespace.views.initializeDatabaseView) {
                    namespace.views.initializeDatabaseView(); // Reload with new time filter
                }
            });
        }
        
        // Database search input
        const dbSearch = document.getElementById('db-search');
        if (dbSearch) {
            dbSearch.addEventListener('input', () => {
                if (namespace.views && namespace.views.renderDatabaseDocuments) {
                    namespace.views.renderDatabaseDocuments(); // Re-render with search filter
                }
            });
        }
        
        // Database sort select
        const dbSort = document.getElementById('db-sort');
        if (dbSort) {
            dbSort.addEventListener('change', () => {
                if (namespace.views && namespace.views.renderDatabaseDocuments) {
                    namespace.views.renderDatabaseDocuments(); // Re-render with new sort order
                }
            });
        }
        
        // Refresh database button
        const refreshDatabase = document.getElementById('refresh-database');
        if (refreshDatabase) {
            refreshDatabase.addEventListener('click', () => {
                if (namespace.views && namespace.views.initializeDatabaseView) {
                    namespace.views.initializeDatabaseView(); // Reload database data
                }
            });
        }
        
        // Semantic search button
        const semanticSearchButton = document.getElementById('semantic-search-button');
        if (semanticSearchButton) {
            semanticSearchButton.addEventListener('click', () => {
                if (namespace.views && namespace.views.performSemanticSearch) {
                    namespace.views.performSemanticSearch();
                }
            });
        }
        
        // Semantic search input (enter key)
        const semanticSearchInput = document.getElementById('semantic-search-input');
        if (semanticSearchInput) {
            semanticSearchInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && namespace.views && namespace.views.performSemanticSearch) {
                    namespace.views.performSemanticSearch();
                }
            });
        }
        
        // Refresh recommendations button
        const refreshRecommendations = document.getElementById('refresh-recommendations');
        if (refreshRecommendations) {
            refreshRecommendations.addEventListener('click', () => {
                if (namespace.views && namespace.views.initializeRecommendationsView) {
                    namespace.views.initializeRecommendationsView(); // Reload recommendations
                }
            });
        }
        
        // Re-train model button
        const retrainModel = document.getElementById('retrain-model');
        if (retrainModel) {
            retrainModel.addEventListener('click', () => {
                if (namespace.views && namespace.views.bootstrapRecommender) {
                    namespace.views.bootstrapRecommender(); // Force re-training of the model
                }
            });
        }
        
        // Register for progress updates is now handled at the module level
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
            await eel.save_ui_config(newConfig)();
            state.config = newConfig;
            showAlert('Configuration saved successfully.', 'success');
        } catch (error) {
            showAlert(`Error saving configuration: ${error}`, 'danger');
        } finally {
            showLoading(false);
        }
    }

    // Export public functions and objects
    namespace.core = {
        state,
        initializeApp,
        navigateTo,
        loadConfig,
        saveConfig,
        setupEventListeners,
        updateConfigDisplay
    };

    // Initialize the application when the DOM is loaded
    document.addEventListener('DOMContentLoaded', () => {
        initializeApp();
    });

    // Expose necessary functions to Python via Eel
    if (typeof eel !== 'undefined') {
        eel.expose(showAlert);
        eel.expose(showLoading);
        eel.expose(function updateProgress(current, total, description) {
            if (components.updateProgress) {
                components.updateProgress(current, total, description);
            }
        });
    }
})(window.PaperRecommender);