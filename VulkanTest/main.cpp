#define GLFW_INCLUDE_VULKAN
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define STB_IMAGE_IMPLEMENTATION
#define TINYOBJLOADER_IMPLEMENTATION
#define GLM_ENABLE_EXPERIMENTAL

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <stb_image.h>
#include <tiny_obj_loader.h>

#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <optional>
#include <set>
#include <cstdint>	
#include <algorithm>
#include <fstream>
#include <array>
#include <chrono>
#include <unordered_map>
#include <glm/gtx/hash.hpp>

#ifdef NDEBUG
	const bool enableValidationLayers = false;
#else
	const bool enableValidationLayers = true;
#endif

#pragma region Declaración de constantes globales

const uint32_t WIDTH = 1920;	 //El ancho de la ventana (window)
const uint32_t HEIGHT = 1080;	 //El alto de la ventana (window)

const std::string MODEL_PATH = "chest_low_Final.obj";
const std::string TEXTURE_PATH = "chest_Base_Color.jpg";

const std::vector<const char*> validationLayers = { "VK_LAYER_KHRONOS_validation" };	//Colección de capas de validación que utilizamos para configurar el debugger
const std::vector<const char*> deviceExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };  //Colección de extensiones del dispositivo (físico)

const int MAX_FRAMES_IN_FLIGHT = 2;

#pragma endregion

#pragma region Funciones de ámbito global

	/*
		Tenemos que declarar las siguientes funciones proxy para crear y destruir el objeto debug messenger.
		Ya que al tratarse de funciones extendidas, no se cargan de forma automática.
		Buscaremos las direcciones llamando a la función vkGetInstanceProcAddr
	*/

	VkResult CreateDebugUtilsMessengerEXT(VkInstance instance,
										const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
										const VkAllocationCallbacks* pAllocator,
										VkDebugUtilsMessengerEXT* pDebugMessenger)
	{
		auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");

		if (func != nullptr)
		{
			return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
		}
		else
		{
			return VK_ERROR_EXTENSION_NOT_PRESENT;
		}
	}

	void DestroyDebugUtilsMessengerEXT(VkInstance instance,
										VkDebugUtilsMessengerEXT debugMessenger,
										const VkAllocationCallbacks* pAllocator)
	{
		auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");

		if (func != nullptr)
		{
			func(instance, debugMessenger, pAllocator);
		}
	}

	static std::vector<char> readFile(const std::string& filename)
	{
		/*
			ate: at the end of the file
			binary: leer fichero como fichero binario
		*/
		std::ifstream file(filename, std::ios::ate | std::ios::binary);

		if (!file.is_open())
		{
			throw std::runtime_error("failed to open file!");
		}

		//Creamos un buffer del tamaño del fichero
		size_t fileSize = (size_t)file.tellg();
		std::vector<char> buffer(fileSize);
		//Leemos los bytes hasta el principio del fichero
		file.seekg(0);
		file.read(buffer.data(), fileSize);

		file.close();
		return buffer;
	}

#pragma endregion

#pragma region Definición de estructuras

	struct QueueFamilyIndices
	{
		std::optional<uint32_t> graphicsFamily;
		std::optional<uint32_t> presentFamily;

		bool isComplete()
		{
			return graphicsFamily.has_value() && presentFamily.has_value();
		}
	};

	struct SwapChainSupportDetails
	{
		VkSurfaceCapabilitiesKHR capabilities;
		std::vector<VkSurfaceFormatKHR> formats;
		std::vector<VkPresentModeKHR> presentModes;
	};

	struct Vertex
	{
		glm::vec3 pos;
		glm::vec3 color;
		glm::vec2 texCoord;

		static VkVertexInputBindingDescription getBindingDescription()
		{
			/*
				VertexInputBindingDescriptión indica a  qué ritmo cargar los datos de los vértices desde la memoria.

				binding: indica el índice en el array de bindings (en nuestro caso solamente tenemos uno)

				stride: indica el número de bytes desde una entrada a la siguiente.

				inputRate puede tener los siguientes valores:

				• VK_VERTEX_INPUT_RATE_VERTEX: Move to the next data entry after each vertex

				• VK_VERTEX_INPUT_RATE_INSTANCE: Move to the next data entry after each instance
			*/
			VkVertexInputBindingDescription bindingDescription{};
			bindingDescription.binding = 0;
			bindingDescription.stride = sizeof(Vertex);
			bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

			return bindingDescription;
		}

		static std::array<VkVertexInputAttributeDescription, 3>	getAttributeDescriptions()
		{
			/*
				VertexInputAttributeDescription indica cómo extraer un atributo de un vértice, de los datos de los vértices descritos en el binding description.

				binding: comunica a Vulkan de que binding vienen los datos por vértice.

				location: referencia la directiva location del input en el vertex shader.

				format: describe el tipo de dato para el atributo. Ha de describirse de igual manera que los formatos de color:

					• float: VK_FORMAT_R32_SFLOAT
					• vec2: VK_FORMAT_R32G32_SFLOAT
					• vec3: VK_FORMAT_R32G32B32_SFLOAT
					• vec4: VK_FORMAT_R32G32B32A32_SFLOAT

					El número de canales de color debe coincidir con el número de componentes en el tipo de dato del shader.
					El tipo del color (SFLOAT, UINT, SINT) y el ancho de bits también deben coincidir con el tipo definido en el shader.

					Ejemplos:
					• ivec2: VK_FORMAT_R32G32_SINT, a 2-component vector of 32-bit signed integers
					• uvec4: VK_FORMAT_R32G32B32A32_UINT, a 4-component vector of 32-bit unsigned integers
					• double: VK_FORMAT_R64_SFLOAT, a double-precision (64-bit) float

				offset: Especifica el número de bytes que se leerán desde el inicio de los datos por vértice.
			*/
			std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions{};

			/*Posición de los vértices*/
			attributeDescriptions[0].binding = 0;
			attributeDescriptions[0].location = 0;
			attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
			attributeDescriptions[0].offset = offsetof(Vertex, pos);
			/*Color de los vértices*/
			attributeDescriptions[1].binding = 0;
			attributeDescriptions[1].location = 1;
			attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
			attributeDescriptions[1].offset = offsetof(Vertex, color);
			/*Coordenadas de las texturas*/
			attributeDescriptions[2].binding = 0;
			attributeDescriptions[2].location = 2;
			attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
			attributeDescriptions[2].offset = offsetof(Vertex, texCoord);

			return attributeDescriptions;
		}

		bool operator==(const Vertex& other) const
		{
			return pos == other.pos && color == other.color && texCoord ==	other.texCoord;
		}
	};

	namespace std {
		template<> struct hash<Vertex>
		{ 
			size_t operator()(Vertex const& vertex) const 
			{
				return ((hash<glm::vec3>()(vertex.pos) ^ (hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^ (hash<glm::vec2>()(vertex.texCoord) << 1);
			}			
		};
	}

	struct UniformBufferObject
	{
		glm::mat4 model;
		glm::mat4 view;
		glm::mat4 proj;
	};

#pragma endregion

class HelloTriangleApplication
{
	public:

		void run()
		{
			initWindow();
			initVulkan();
			mainLoop();
			cleanup();
		}

	private:

	#pragma region Miembros privados de la clase

		GLFWwindow* window;

		VkInstance instance;

		VkDebugUtilsMessengerEXT debugMessenger;

		VkSurfaceKHR surface;

		//Dispositivos y colas
		VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
		VkDevice device;
		VkQueue graphicsQueue;
		VkQueue presentQueue;

		//Swap chain
		VkSwapchainKHR swapChain;
		std::vector<VkImage> swapChainImages;
		VkFormat swapChainImageFormat;
		VkExtent2D swapChainExtent;
		std::vector<VkImageView> swapChainImageViews;

		//Pipeline de gráficos
		VkRenderPass renderPass;
		VkDescriptorSetLayout descriptorSetLayout;
		VkPipelineLayout pipelineLayout;
		VkPipeline graphicsPipeline;

		//Framebuffers
		std::vector<VkFramebuffer> swapChainFramebuffers;
		bool framebufferResized = false;

		//Comandos
		VkCommandPool commandPool;
		std::vector<VkCommandBuffer> commandBuffers;

		//Objetos de sincronización: semáforos y vallas (semaphores and fences)
		std::vector<VkSemaphore> imageAvailableSemaphores;
		std::vector<VkSemaphore> renderFinishedSemaphores;
		std::vector<VkFence> inFlightFences;
		std::vector<VkFence> imagesInFlight;
		size_t currentFrame = 0;

		//Vertex/Index data
		std::vector<Vertex> vertices;
		std::vector<uint32_t> indices;

		//Buffers
		VkBuffer vertexBuffer;
		VkDeviceMemory vertexBufferMemory;
		VkBuffer indexBuffer;
		VkDeviceMemory indexBufferMemory;
		std::vector<VkBuffer> uniformBuffers;
		std::vector<VkDeviceMemory> uniformBuffersMemory;

		//Descriptores
		VkDescriptorPool descriptorPool;
		std::vector<VkDescriptorSet> descriptorSets;

		//Imágenes
		uint32_t mipLevels;
		VkImage textureImage;
		VkDeviceMemory textureImageMemory;
		VkImageView textureImageView;
		
		//Samplers
		VkSampler textureSampler;

		//Depth image
		VkImage depthImage;
		VkDeviceMemory depthImageMemory;
		VkImageView depthImageView;

		//Multimuestreo
		VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;
		VkImage colorImage;
		VkDeviceMemory colorImageMemory;
		VkImageView colorImageView;

	#pragma endregion

	#pragma region Funciones privadas de la clase

	#pragma region Funciones generales

		void initWindow()
		{
			glfwInit();

			glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); //Sin API (GLFW)
			//glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);   //Indicamos si la ventana es redimensionable

			window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
			glfwSetWindowUserPointer(window, this);
			glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
		}

		void createSurface()
		{
			if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create window surface!");
			}
		}

		void initVulkan()
		{
			createInstance();
			setupDebugMessenger();
			createSurface();
			pickPhysicalDevice();
			createLogicalDevice();
			createSwapChain();
			createImageViews();
			createRenderPass();
			createDescriptorSetLayout();
			createGraphicsPipeline();
			createCommandPool();
			createColorResources();
			createDepthResources();
			createFramebuffers();
			createTextureImage();
			createTextureImageView();
			createTextureSampler();
			loadModel();
			createVertexBuffer();
			createIndexBuffer();
			createUniformBuffers();
			createDescriptorPool();
			createDescriptorSets();
			createCommandBuffers();
			createSyncObjects();
		}

		void mainLoop()
		{
			while (!glfwWindowShouldClose(window))
			{
				glfwPollEvents();
				drawFrame();
			}

			vkDeviceWaitIdle(device);
		}

		void cleanup()
		{
			//Destruimos todos los objectos que deben ser destruidos de forma explícita al finalizar la ejecución
			cleanupSwapChain();

			vkDestroySampler(device, textureSampler, nullptr);

			vkDestroyImageView(device, textureImageView, nullptr);
			vkDestroyImage(device, textureImage, nullptr);
			vkFreeMemory(device, textureImageMemory, nullptr);

			vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

			vkDestroyBuffer(device, indexBuffer, nullptr);
			vkFreeMemory(device, indexBufferMemory, nullptr);

			vkDestroyBuffer(device, vertexBuffer, nullptr);
			vkFreeMemory(device, vertexBufferMemory, nullptr);
				
			for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
			{
				vkDestroySemaphore(device, renderFinishedSemaphores[i],	nullptr);
				vkDestroySemaphore(device, imageAvailableSemaphores[i],	nullptr);
				vkDestroyFence(device, inFlightFences[i], nullptr);
			}

			vkDestroyCommandPool(device, commandPool, nullptr);					
		
			vkDestroyDevice(device, nullptr);

			if (enableValidationLayers)
			{
				DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
			}

			vkDestroySurfaceKHR(instance, surface, nullptr);
			vkDestroyInstance(instance, nullptr);

			glfwDestroyWindow(window);

			glfwTerminate();
		}

		static void framebufferResizeCallback(GLFWwindow* window, int width, int height)
		{
			auto app =	reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
			app->framebufferResized = true;
		}

	#pragma endregion

	#pragma region Renderizado

		void drawFrame()
		{
			/*
				Esperamos a que el frame previo finalice.

				Indicamos UINT64_MAX (máximo valor de un entero sin signo de 64 bits) para inhabilitar el parámetro timeout
			*/
			vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
		
			/*
				Obtenemos la siguiente imagen.

				Indicamos UINT64_MAX (máximo valor de un entero sin signo de 64 bits) para inhabilitar el parámetro timeout
				Indicamos imageIndex, para hacer referencia al objeto VkImage en nuestro array de swapChainImages. De forma que elijamos el command buffer adecuado.
			*/
			uint32_t imageIndex;
			VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

			/*
				Comprobamos si la recreación del swap chain es necesaria. Para ello, evaluamos los siguientes valores de salida para las funciones vkAcquireNextImageKHR y vkQueuePresentKHR: 

				• VK_ERROR_OUT_OF_DATE_KHR: The swap chain has become incompatible with the surface and can no longer be used for rendering. Usually happens after a window resize.

				• VK_SUBOPTIMAL_KHR: The swap chain can still be used to successfully present to the surface, but the surface properties are no longer matched exactly.
			*/
			if (result == VK_ERROR_OUT_OF_DATE_KHR)
			{
				recreateSwapChain();
				return;//Retornamos tras recrear el swap chain para intentar obtener la imagen en el siguiente frame
			}
			else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
			{
				throw std::runtime_error("failed to acquire swap chain image!");
			}

			//Comprobamos si un frame previo esta utilizando esta imagen
			if (imagesInFlight[imageIndex] != VK_NULL_HANDLE)
			{
				vkWaitForFences(device, 1, &imagesInFlight[imageIndex],	VK_TRUE, UINT64_MAX);
			}
			//Marcar que la imagen está siendo usada por el frame actual
			imagesInFlight[imageIndex] = inFlightFences[currentFrame];

			//Actualizamos el buffer de nuestro UBO
			updateUniformBuffer(imageIndex);

			VkSubmitInfo submitInfo{};
			submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		
			/*
				Indicamos con qué semáforo queremos esperar y en que etapa del pipeline. Nos interesa esperar a dibujar los colores de la imagen hasta que ésta misma esté disponible. 
			*/
			VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
			VkPipelineStageFlags waitStages[] =	{ VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
			submitInfo.waitSemaphoreCount = 1;
			submitInfo.pWaitSemaphores = waitSemaphores;
			submitInfo.pWaitDstStageMask = waitStages;

			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers = &commandBuffers[imageIndex];

			/*
				Indicamos qué semáforos señalizamos, una vez finalizada la ejecución del command buffer.
			*/
			VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] };
			submitInfo.signalSemaphoreCount = 1;
			submitInfo.pSignalSemaphores = signalSemaphores;

			vkResetFences(device, 1, &inFlightFences[currentFrame]);

			if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to submit draw command buffer!");
			}

			VkPresentInfoKHR presentInfo{};
			presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
			presentInfo.waitSemaphoreCount = 1;
			presentInfo.pWaitSemaphores = signalSemaphores;

			VkSwapchainKHR swapChains[] = { swapChain };
			presentInfo.swapchainCount = 1;
			presentInfo.pSwapchains = swapChains;
			presentInfo.pImageIndices = &imageIndex;
			presentInfo.pResults = nullptr; // Optional

			/*
				Enviamos la solicitud para presentar una imagen al swap chain
			*/
			result = vkQueuePresentKHR(presentQueue, &presentInfo);

			/*
				Comprobamos si la recreación del swap chain es necesaria. Para ello, evaluamos los siguientes valores de salida para las funciones vkAcquireNextImageKHR y vkQueuePresentKHR:

				• VK_ERROR_OUT_OF_DATE_KHR: The swap chain has become incompatible with the surface and can no longer be used for rendering. Usually happens after a window resize.

				• VK_SUBOPTIMAL_KHR: The swap chain can still be used to successfully present to the surface, but the surface properties are no longer matched exactly.
			*/
			if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized)
			{
				framebufferResized = false;
				recreateSwapChain();
			}
			else if (result != VK_SUCCESS)
			{
				throw std::runtime_error("failed to present swap chain image!");
			}

			vkQueueWaitIdle(presentQueue);

			//Recalculamos el frame actual
			currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
		}

		/*
			Los eventos necesarios para dibujar un frame se ejecutan mediante llamadas a funciones. El problema es que se ejecutan de manera asíncrona.
			Es por ello que necesitamos una forma para controlar cúando realizar la llamada a una función y cúando esperar a que otra llamada haya finalizado.

			Existen dos objetos que permiten coordinar operaciones de forma síncrona: vallas (fences) y semáforos (semaphores).

			Los semáforos se utilizan mayormente para sincronizar operaciones entre o durante colas de comandos, se pueden señalizar y esperar a su señal.
			Las vallas se utilizan de manera similar, con la diferencia de que las controlamos en nuestro código.
		*/
		void createSyncObjects()
		{
			imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
			renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
			inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
			imagesInFlight.resize(swapChainImages.size(), VK_NULL_HANDLE);

			VkSemaphoreCreateInfo semaphoreInfo{};
			semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

			VkFenceCreateInfo fenceInfo{};
			fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
			//Es necesario inicializarlo como señalizado. Si no, la función vkWaitForFences esperará indefinidamente antes de dibujar el primer frame
			fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

			for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
			{
				if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
					vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
					vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS)
				{
					throw std::runtime_error("failed to create synchronization objects for a frame!");
				}
			}
		}

	#pragma endregion

	#pragma region Instancia

		void createInstance()
		{
			if (enableValidationLayers && !checkValidationLayerSupport())
			{
				throw std::runtime_error("validation layers requested but not available!");
			}

			//Información de la aplicación
			VkApplicationInfo appInfo{};
			appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
			appInfo.pApplicationName = "Hello Triangle";
			appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
			appInfo.pEngineName = "No engine";
			appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
			appInfo.apiVersion = VK_API_VERSION_1_0;

			//Información de la instancia
			VkInstanceCreateInfo	createInfo{};
			createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
			createInfo.pApplicationInfo = &appInfo;

			//Extensiones requeridas por la instancia
			auto glfwExtensions = getRequiredExtensions();
			createInfo.enabledExtensionCount = static_cast<uint32_t>(glfwExtensions.size());
			createInfo.ppEnabledExtensionNames = glfwExtensions.data();

			if (enableValidationLayers)
			{
				createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
				createInfo.ppEnabledLayerNames = validationLayers.data();
			}
			else
			{
				createInfo.enabledLayerCount = 0;
			}

			//Imprimir extensiones requeridas
			//printRequiredExtensions(createInfo);

			//Extensiones disponibles
			uint32_t extensionCount = 0;
			vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

			std::vector<VkExtensionProperties> extensions(extensionCount);
			vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());

			//Imprimir extensiones disponibles
			//printAvailableExtensions(extensions);

			//Creamos un debug messenger adicional para utilizarlo durante vkCreateInstance y vkDestroyInstance
			VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo;

			if (enableValidationLayers)
			{
				createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
				createInfo.ppEnabledLayerNames = validationLayers.data();

				populateDebugMessengerCreateInfo(debugCreateInfo);
				createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
			}
			else
			{
				createInfo.enabledLayerCount = 0;

				createInfo.pNext = nullptr;
			}

			//Creamos la instancia
			VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);

			if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create instance!");
			}
		}

		void printRequiredExtensions(const VkInstanceCreateInfo& createInfo)
		{
			std::cout << "Required extensions:\n";

			std::cout << createInfo.ppEnabledExtensionNames << std::endl;
		}

		void printAvailableExtensions(const std::vector<VkExtensionProperties>& extensions)
		{
			std::cout << "Available extensions:\n";

			for (const auto& extension : extensions)
			{
				std::cout << '\t' << extension.extensionName << '\n';
			}
		}

		bool checkValidationLayerSupport()
		{
			//Comprobamos las capas de validación soportadas
			uint32_t layerCount = 0;
			vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

			std::vector<VkLayerProperties> availableLayers(layerCount);
			vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

			// Imprimir validationLayers disponibles
			//printValidationLayerSupport(availableLayers);

			for (const char* layerName : validationLayers)
			{
				bool layerFound = false;

				for (const auto& layerProperties : availableLayers)
				{
					if (strcmp(layerName, layerProperties.layerName) == 0)
					{
						layerFound = true;
						break;
					}
				}

				if (!layerFound)
				{
					return false;
				}
			}

			return true;
		}

		void printValidationLayerSupport(const std::vector<VkLayerProperties>& availableLayers)
		{
			std::cout << "Available layers:\n";

			for (const char* layerName : validationLayers)
			{
				for (const auto& layerProperties : availableLayers)
				{
					std::cout << '\t' << layerProperties.layerName << std::endl;
				}

			}
		}

		std::vector<const char*> getRequiredExtensions()
		{
			uint32_t glfwExtensionCount = 0;
			const char** glfwExtensions;

			glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

			std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

			if (enableValidationLayers) {
				extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
			}

			return extensions;
		}

	#pragma endregion

	#pragma region Debug messenger

		/*
			Mensaje del debugger
			--------------------

			messageSeverity: Indica la severidad del mensaje según la enumeración siguiente:

				• VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT: Diagnosticmessage
				• VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT: Informational message like the creation of a resource
				• VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT: Message about behavior that is not necessarily an error, but very likely a bug in your application
				• VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT: Message about behavior that is invalid and may cause crashes

			messageType: Indica el tipo de mensaje según la enumeración siguiente:

				• VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT: Some event has happened that is unrelated to the specification or performance
				• VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT: Something has	happened that violates the specification or indicates a possible mistake
				• VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT: Potential non optimal use of Vulkan

			pCallBackData: se refiere a la estructura VkDebugUtilsMessengerCallbackDataEXT, que contiene los detalles del mensaje:

				• pMessage: The debug message as a null-terminated string
				• pObjects: Array of Vulkan object handles related to the message
				• objectCount: Number of objects in array
				...

			pUserData: Contiene un puntero que fue especificado durante la configuración del CallBack y nos permite pasarle nuestros propios datos
		*/

		static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
															VkDebugUtilsMessageTypeFlagsEXT messageType,
															const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
															void* pUserData)
		{
			std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

			return VK_FALSE;
		}

		void setupDebugMessenger()
		{
			if (!enableValidationLayers) return;

			VkDebugUtilsMessengerCreateInfoEXT createInfo;
			populateDebugMessengerCreateInfo(createInfo);

			if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to set up debug messenger!");
			}
		}

		void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo)
		{
			createInfo = {};
			createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
			createInfo.messageSeverity =
				//VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
				VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
				VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
			createInfo.messageType =
				VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
				VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
				VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
			createInfo.pfnUserCallback = debugCallback;
		}

	#pragma endregion

	#pragma region Dispositivo (físico)

		void pickPhysicalDevice()
		{
			uint32_t deviceCount = 0;
			vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

			if (deviceCount == 0)
			{
				throw std::runtime_error("failed to find GPUs with Vulkan support!");
			}

			std::vector<VkPhysicalDevice> devices(deviceCount);
			vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

			for (const auto& device : devices)
			{
				if (isDeviceSuitable(device))
				{
					physicalDevice = device;
					msaaSamples = getMaxUsableSampleCount();
					break;
				}
			}

			if (physicalDevice == VK_NULL_HANDLE)
			{
				throw std::runtime_error("failed to find a suitable GPU!");
			}
		}

		bool isDeviceSuitable(VkPhysicalDevice device)
		{
			QueueFamilyIndices indices = findQueueFamilies(device);

			bool extensionsSupported = checkDeviceExtensionSupport(device);
		
			bool swapChainAdequate = false;

			if (extensionsSupported) {
				SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
				swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
			}

			VkPhysicalDeviceFeatures supportedFeatures;
			vkGetPhysicalDeviceFeatures(device, &supportedFeatures);

			return indices.isComplete() && extensionsSupported && swapChainAdequate && supportedFeatures.samplerAnisotropy;
		}

		bool checkDeviceExtensionSupport(VkPhysicalDevice device)
		{
			//Comprobamos si el dispositivo (físico) soporta todas las extensiones disponibles
			uint32_t extensionCount;
			vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
		
			std::vector<VkExtensionProperties> availableExtensions(extensionCount);
			vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

			std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

			for (const auto& extension : availableExtensions)
			{
				requiredExtensions.erase(extension.extensionName);
			}

			return requiredExtensions.empty();
		}

		QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device)
		{
			QueueFamilyIndices indices;

			uint32_t queueFamilyCount = 0;
			vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

			std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
			vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

			int i = 0;

			for (const auto& queueFamily : queueFamilies)
			{
				//Familias que tienen soporte para comandos que dibujan gráficos
				if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)
				{
					indices.graphicsFamily = i;
				}

				VkBool32 presentSupport = false;
				vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

				//Familias que tienen soporte para presentar gráficos en una superficie
				if (presentSupport)
				{
					indices.presentFamily = i;
				}
			
				if (indices.isComplete())
				{
					break;
				}

				i++;
			}

			return indices;
		}

	#pragma endregion

	#pragma region Dispositivo (lógico)

		void createLogicalDevice()
		{
			//Generamos los queueCreateInfos para cada familia
			QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

			std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
			std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.graphicsFamily.value() };

			//Prioridad de cola, es necesario incluso cuando solamente existe una única cola
			float queuePriority = 1.0f;
			for (uint32_t queueFamily : uniqueQueueFamilies)
			{
				VkDeviceQueueCreateInfo queueCreateInfo{};
				queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
				queueCreateInfo.queueFamilyIndex = queueFamily;
				queueCreateInfo.queueCount = 1;
				queueCreateInfo.pQueuePriorities = &queuePriority;
				queueCreateInfos.push_back(queueCreateInfo);
			}

			VkPhysicalDeviceFeatures deviceFeatures{};
			deviceFeatures.samplerAnisotropy = VK_TRUE;

			//Información del dispositivo (lógico)
			VkDeviceCreateInfo createInfo{};
			createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
			createInfo.pQueueCreateInfos = queueCreateInfos.data();
			createInfo.queueCreateInfoCount = static_cast<uint32_t> (queueCreateInfos.size());
			createInfo.pEnabledFeatures = &deviceFeatures;
			createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
			createInfo.ppEnabledExtensionNames = deviceExtensions.data();

			if (enableValidationLayers) {
				createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
				createInfo.ppEnabledLayerNames = validationLayers.data();
			}
			else
			{
				createInfo.enabledLayerCount = 0;
			}

			//Creamos el dispositivo
			if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create logical device!");
			}

			/*Obtenemos el handler para cada queue family.
			En este caso, como solamente tenemos un único queue por cada familia utilizamos el índice 0*/
			vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
			vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
		}

	#pragma endregion

	#pragma region Swap chain

		SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device)
		{
			SwapChainSupportDetails details;

			vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

			uint32_t formatCount;
			vkGetPhysicalDeviceSurfaceFormatsKHR(device,surface,&formatCount,nullptr);

			if (formatCount != 0)
			{
				details.formats.resize(formatCount);
				vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
			}

			uint32_t presentModeCount;
			vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

			if (presentModeCount != 0)
			{
				details.presentModes.resize(presentModeCount);
				vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &formatCount, details.presentModes.data());
			}

			return details;
		}

		VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats)
		{
			//Comprobamos si existe el formato B8G8R8_SRGB y el espacio de color es el estandar SRGB, si no es así se devuelve el primer formato disponible
			for (const auto& availableFormat : availableFormats)
			{
				if (availableFormat.format == VK_FORMAT_B8G8R8_SRGB && availableFormat.colorSpace == VK_COLORSPACE_SRGB_NONLINEAR_KHR)
				{
					return availableFormat;
				}
			}

			return availableFormats[0];
		}

		/*
			Modos de presentación
			---------------------

			• VK_PRESENT_MODE_IMMEDIATE_KHR: Images submitted by your application are transferred to the screen right away, which may result in tearing.

			• VK_PRESENT_MODE_FIFO_KHR: The swap chain is a queue where the display takes an image from the front of the queue when the display is refreshed and the program inserts rendered images at the back of the queue.
			If the queue is full then the program has to wait. This is most similar to vertical sync as found in modern games. The moment that the display is refreshed is known as “vertical blank”.

			• VK_PRESENT_MODE_FIFO_RELAXED_KHR: This mode only differs from the previous one if the application is late and the queue was empty at the last vertical blank.
			Instead of waiting for the next vertical blank, the image is transferred right away when it finally arrives. This may result in visible tearing.

			• VK_PRESENT_MODE_MAILBOX_KHR: This is another variation of the second mode. Instead of blocking the application when the queue is full, the images that are already queued are simply replaced with the newer ones.
			This mode can be used to implement triple buffering, which allows you to avoid tearing with significantly less latency issues than standard vertical sync that uses double buffering.
		*/
		VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes)
		{
			//Comprobamos si existe el modo de presentación MAILBOX (triple buffering), si no es así se devuelve el modo FIFO
			for (const auto& availablePresentMode : availablePresentModes)
			{
				if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR)
				{
					return availablePresentMode;
				}
			}

			return VK_PRESENT_MODE_FIFO_KHR;
		}

		VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities)
		{
			if (capabilities.currentExtent.width != UINT32_MAX)
			{
				return capabilities.currentExtent;
			}
			else
			{
				int width, height;
				glfwGetFramebufferSize(window, &width, &height);

				//Si la extensión supera la dimensión permitida en las capacidades de la superficie, la ajustamos para que se adecue a las mismas
				VkExtent2D actualExtent = { static_cast<uint32_t>(width), static_cast<uint32_t>(height) };
				actualExtent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actualExtent.width));
				actualExtent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height,	actualExtent.height));
			
				return actualExtent;
			}
		}

		void createSwapChain()
		{
			//Obtenemos la información necesaria para crear el swap chain
			SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);
			VkSurfaceFormatKHR surfaceFormat =	chooseSwapSurfaceFormat(swapChainSupport.formats);
			VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
			VkExtent2D extent =	chooseSwapExtent(swapChainSupport.capabilities);

			//En algunas situaciones tendremos que esperar a que el driver finalice alguna operación antes de poder obtener otra imagen, por ello obtenemos una imagen más que el mínimo necesario
			uint32_t imageCount = swapChainSupport.capabilities.minImageCount +	1; 

			//Comprobamos que el número de imágenes no supere el máximo soportado por el swap chain
			if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount >	swapChainSupport.capabilities.maxImageCount)
			{
				imageCount = swapChainSupport.capabilities.maxImageCount;
			}

			VkSwapchainCreateInfoKHR createInfo{};
			createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
			createInfo.surface = surface;
			createInfo.minImageCount = imageCount;
			createInfo.imageFormat = surfaceFormat.format;
			createInfo.imageColorSpace = surfaceFormat.colorSpace;
			createInfo.imageExtent = extent;
			//Cantidad de capas que componen la imagen, el valor siempre es 1 (excepto si se desarrolla una aplicación que soporte 3D stereoscopico)
			createInfo.imageArrayLayers = 1;
			//Si vamos a renderizar la imagen a una imagen separada primero (post-procesado) podemos utilizar el valor VK_IMAGE_USAGE_TRANSFER_DST_BIT (después renderizaremos la imagen desde la memoria)
			createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT; 

			QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
			uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

			/* Si la cola de gráficos (graphics queue) y la cola de presentación (present queue) son diferentes. Existen dos maneras de gestionar el acceso de ambas colas a las imágenes:
			
				• VK_SHARING_MODE_EXCLUSIVE: An image is owned by one queue family at a time and ownership must be explicitly transfered before using it in another queue family. This option offers the best performance.

				• VK_SHARING_MODE_CONCURRENT: Images can be used across multiple queue families without explicit ownership transfers.

			*/
			if (indices.graphicsFamily != indices.presentFamily)
			{
				createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
				createInfo.queueFamilyIndexCount = 2;
				createInfo.pQueueFamilyIndices = queueFamilyIndices;
			}
			else
			{
				createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
				createInfo.queueFamilyIndexCount = 0; // Optional
				createInfo.pQueueFamilyIndices = nullptr; // Optional
			}

			createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
			createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;//Para ignorar el canal alfa (alpha channel)
			createInfo.presentMode = presentMode;
			createInfo.clipped = VK_TRUE;//Habilitar clipping (para discriminar los pixels ocultos)
			createInfo.oldSwapchain = VK_NULL_HANDLE;

			//Creamos el Swap chain
			if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create swap chain!");
			}

			//Obtenemos los handles para las imágenes
			vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
			swapChainImages.resize(imageCount);
			vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

			//Almacenamos el formato de la superficie y la extensión en las variables miembro para su futuro uso
			swapChainImageFormat = surfaceFormat.format;
			swapChainExtent = extent;
		}

		void cleanupSwapChain()
		{
			//Destruimos los objetos que utiliza el swap chain de forma explícita, para recrearlos a la vez que el swap chain
			vkDestroyImageView(device, colorImageView, nullptr);
			vkDestroyImage(device, colorImage, nullptr);
			vkFreeMemory(device, colorImageMemory, nullptr);

			vkDestroyImageView(device, depthImageView, nullptr);
			vkDestroyImage(device, depthImage, nullptr);
			vkFreeMemory(device, depthImageMemory, nullptr);

			for (auto framebuffer : swapChainFramebuffers)
			{
				vkDestroyFramebuffer(device, framebuffer, nullptr);
			}

			vkFreeCommandBuffers(device, commandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());
		
			vkDestroyPipeline(device, graphicsPipeline, nullptr);
			vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
			vkDestroyRenderPass(device, renderPass, nullptr);

			for (auto imageView : swapChainImageViews)
			{
				vkDestroyImageView(device, imageView, nullptr);
			}

			vkDestroySwapchainKHR(device, swapChain, nullptr);

			for (size_t i = 0; i < swapChainImages.size(); i++)
			{
				vkDestroyBuffer(device, uniformBuffers[i], nullptr);
				vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
			}

			vkDestroyDescriptorPool(device, descriptorPool, nullptr);
		}

		/*
			Se puede dar el caso en el que necesitemos volver a crear el Swap chain. El ejemplo más típico es qué se redimensione la ventana (window) de la aplicación.
			Es por ello que debemos volver a crear todos los objetos que dependen del swap chain.

			Los objetos viewport y scissor se especifican durante la creación del pipeline, por lo que también debemos incluirlo.
		*/
		void recreateSwapChain()
		{
			int width = 0, height = 0;
			glfwGetFramebufferSize(window, &width, &height);

			/*
				Cuando el framebuffer tiene tamaño 0, por ejemplo cuando la ventana está minimizada, pausamos el procesamiento de eventos hasta que la ventana vuelva a estar en primer plano.
			*/
			while (width == 0 || height == 0)
			{
				glfwGetFramebufferSize(window, &width, &height);
				glfwWaitEvents();
			}

			//Esperamos a que se haya terminado cualquier operación en vuelo
			vkDeviceWaitIdle(device);

			cleanupSwapChain();

			createSwapChain();
			createImageViews();
			createRenderPass();
			createGraphicsPipeline();
			createColorResources();
			createDepthResources();
			createFramebuffers();
			createUniformBuffers();
			createDescriptorPool();
			createDescriptorSets();
			createCommandBuffers();
		}

	#pragma endregion

	#pragma region Pipeline de Gráficos

		/*
			El pipeline de los gráficos se compone de diferentes etapas comprendidas entre la lectura de los datos de los vértices mediante uno o varios buffers, hasta la presentación de los fragmentos coloreados al framebuffer.
			Las etapas se dividen en dos tipos: programables y de función fija (fixed-function).

			Podemos programar las etapas shader: vertex, tesselation, geometry y fragment del pipeline. Esto se hace mediante shaders, que se cargan durante la creación del pipeline.
			Los shader definen el comportamiento que va a tener cada una de esas etapas.

			Vulkan utiliza SPIR-V (formato bytecode) como lenguaje shader, sin embargo, el SDK de LunarG incluye un compilador que traduce los shaders desde lenguaje GLSL a SPIR-V.

			El resto de tareas, como por ejemplo el input assembly, la rasterización, viewport o el color blending son gestionados mediante funciones fijas (fixed-functions) que debemos configurar explicitamente.
		*/
		void createGraphicsPipeline()
		{

		#pragma region Etapas programables del pipeline

			auto vertShaderCode = readFile("vert.spv");
			auto fragShaderCode = readFile("frag.spv");

			//Declaramos los shader modules como variables locales, ya que pueden ser destruidos tan pronto como creemos el pipeline
			VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
			VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

			VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
			vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
			vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
			//Indicamos el shaderModule
			vertShaderStageInfo.module = vertShaderModule;
			//La función del shader que invocamos (el entrypoint)
			vertShaderStageInfo.pName = "main";

			VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
			fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
			fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
			//Indicamos el shaderModule
			fragShaderStageInfo.module = fragShaderModule;
			//La función del shader que invocamos (el entrypoint)
			fragShaderStageInfo.pName = "main";

			//Creamos un array con ambas estructuras
			VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

		#pragma endregion
		
		#pragma region Etapas de función fija (fixed-function) del pipeline
			//Vertex input
			auto bindingDescription = Vertex::getBindingDescription();
			auto attributeDescriptions = Vertex::getAttributeDescriptions();

			//Referenciamos los objetos vertexBindingDescriptions y vertexAttributeDescriptions
			VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
			vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
			vertexInputInfo.vertexBindingDescriptionCount = 1;
			vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
			vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
			vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

			/*
				Topologías
				----------
				• VK_PRIMITIVE_TOPOLOGY_POINT_LIST: points from vertices
			
				• VK_PRIMITIVE_TOPOLOGY_LINE_LIST: line from every 2 vertices without reuse
			
				• VK_PRIMITIVE_TOPOLOGY_LINE_STRIP: the end vertex of every line is used as start vertex for the next line

				• VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST: triangle from every 3 vertices without reuse

				• VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP: the second and third vertex of every triangle are used as first two vertices of the next triangle
			*/
			VkPipelineInputAssemblyStateCreateInfo inputAssemblyInfo{};
			inputAssemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
			inputAssemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
			inputAssemblyInfo.primitiveRestartEnable = VK_FALSE;

			/*
				El viewport es la región del framebuffer a la que se va a renderizar la salida
				Lo ajustamos a la dimensión del swapChainExtent
			*/
			VkViewport viewport{};
			viewport.x = 0.0f;
			viewport.y = 0.0f;
			viewport.width = (float)swapChainExtent.width;
			viewport.height = (float)swapChainExtent.height;
			viewport.minDepth = 0.0f;
			viewport.maxDepth = 1.0f;

			/*
				El scissor rectangle "filtra" la región en la que se van a almacenar los pixels. Los pixels fuera del scissor rectangle son descartados por el rasterizador
				Lo ajustamos a la dimensión del swapChainExtent
			*/
			VkRect2D scissor{};
			scissor.offset = { 0, 0 };
			scissor.extent = swapChainExtent;

			//Combinamos el viewport y el scissor rectangle en un viewport state
			VkPipelineViewportStateCreateInfo viewportStateInfo{};
			viewportStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
			viewportStateInfo.viewportCount = 1;
			viewportStateInfo.pViewports = &viewport;
			viewportStateInfo.scissorCount = 1;
			viewportStateInfo.pScissors = &scissor;

			/*
				Modos de polígono
				-----------------
			
				• VK_POLYGON_MODE_FILL: fill the area of the polygon with fragments
			
				• VK_POLYGON_MODE_LINE: polygon edges are drawn as lines
			
				• VK_POLYGON_MODE_POINT: polygon vertices are drawn as points
			*/
			//Rasterizador
			VkPipelineRasterizationStateCreateInfo rasterizer{};
			rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
			rasterizer.depthClampEnable = VK_FALSE;
			rasterizer.rasterizerDiscardEnable = VK_FALSE; //Si el valor es VK_TRUE la geometría nunca pasa a la etapa de rasterización
			rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
			rasterizer.lineWidth = 1.0f;
			rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
			rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; //El eje Y está invertido en la matriz de proyección
			rasterizer.depthBiasEnable = VK_FALSE;
			rasterizer.depthBiasConstantFactor = 0.0f; // Optional
			rasterizer.depthBiasClamp = 0.0f; // Optional
			rasterizer.depthBiasSlopeFactor = 0.0f; // Optional

			//Multimuestreo
			//TODO El multimuestreo se revisitará más adelante
			VkPipelineMultisampleStateCreateInfo multisampling{};
			multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
			multisampling.sampleShadingEnable = VK_FALSE;
			multisampling.rasterizationSamples = msaaSamples;
			multisampling.minSampleShading = 1.0f; // Optional
			multisampling.pSampleMask = nullptr; // Optional
			multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
			multisampling.alphaToOneEnable = VK_FALSE; // Optional

			//Depth stencil
			VkPipelineDepthStencilStateCreateInfo depthStencil{};
			depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
			depthStencil.depthTestEnable = VK_TRUE;	 //Indica si la profundidad del fragmento se compara con el del buffer
			depthStencil.depthWriteEnable = VK_TRUE; //Indica si el la profundidad del fragmento que ha superado el test se escribe en el buffer
			depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
			depthStencil.depthBoundsTestEnable = VK_FALSE;
			depthStencil.minDepthBounds = 0.0f; // Optional
			depthStencil.maxDepthBounds = 1.0f; // Optional
			depthStencil.stencilTestEnable = VK_FALSE;
			
			/*
				Cuando el fragment shader devuelve un color, es necesario combinarlo con el color que está en el framebuffer. Este proceso se denomina color blending.
			
				Existen dos formas de implementarlo:

				• Mezclar el valor antiguo y el nuevo para obtener el color final

				• Combinar los valores en una operación bit a bit
			*/
			//Color blending
			VkPipelineColorBlendAttachmentState colorBlendAttachment{};
			colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
													VK_COLOR_COMPONENT_G_BIT |
													VK_COLOR_COMPONENT_B_BIT |
													VK_COLOR_COMPONENT_A_BIT;
			colorBlendAttachment.blendEnable = VK_FALSE;
			colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
			colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
			colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD; // Optional
			colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
			colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; //	Optional
			colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD; // Optional

			VkPipelineColorBlendStateCreateInfo colorBlending{};
			colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
			colorBlending.logicOpEnable = VK_FALSE;
			colorBlending.logicOp = VK_LOGIC_OP_COPY; // Optional
			colorBlending.attachmentCount = 1;
			colorBlending.pAttachments = &colorBlendAttachment;
			colorBlending.blendConstants[0] = 0.0f; // Optional
			colorBlending.blendConstants[1] = 0.0f; // Optional
			colorBlending.blendConstants[2] = 0.0f; // Optional
			colorBlending.blendConstants[3] = 0.0f; // Optional

		#pragma endregion

		#pragma region Creación del pipeline

			//Pipeline layout
			VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
			pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
			pipelineLayoutInfo.setLayoutCount = 1;
			pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
			pipelineLayoutInfo.pushConstantRangeCount = 0; // Optional
			pipelineLayoutInfo.pPushConstantRanges = nullptr; // Optional

			if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create pipeline layout!");
			}

			VkGraphicsPipelineCreateInfo pipelineInfo{};
			pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		
			//Referenciamos las etapas programables (shaders)
			pipelineInfo.stageCount = 2;
			pipelineInfo.pStages = shaderStages;
		
			//Referenciamos las etapas de función fija
			pipelineInfo.pVertexInputState = &vertexInputInfo;
			pipelineInfo.pInputAssemblyState = &inputAssemblyInfo;
			pipelineInfo.pViewportState = &viewportStateInfo;
			pipelineInfo.pRasterizationState = &rasterizer;
			pipelineInfo.pMultisampleState = &multisampling;
			pipelineInfo.pDepthStencilState = &depthStencil;
			pipelineInfo.pColorBlendState = &colorBlending;
			pipelineInfo.pDynamicState = nullptr; // Optional
		
			pipelineInfo.layout = pipelineLayout;
		
			//Render pass
			pipelineInfo.renderPass = renderPass;
			pipelineInfo.subpass = 0;

			/*
				Es posible crear pipelines derivados de un pipeline existente.
				Para ello también tenemos que especificar VK_PIPELINE_CREATE_DERIVATIVE_BIT.
			*/
			pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // Optional
			pipelineInfo.basePipelineIndex = -1; // Optional

			if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create graphics pipeline!");
			}
		
			vkDestroyShaderModule(device, vertShaderModule, nullptr);
			vkDestroyShaderModule(device, fragShaderModule, nullptr);

		#pragma endregion

		}

		VkShaderModule createShaderModule(const std::vector<char>& code)
		{
			VkShaderModuleCreateInfo createInfo{};
			createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
			createInfo.codeSize = code.size();
			createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

			VkShaderModule shaderModule;

			if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create shader module!");
			}

			return shaderModule;
		}

	#pragma endregion

	#pragma region Render pass

		void createRenderPass()
		{
			/*Color de la imagen*/
			VkAttachmentDescription colorAttachment{};
			colorAttachment.format = swapChainImageFormat;
			colorAttachment.samples = msaaSamples;
			/*
				loadOp
				------

				• VK_ATTACHMENT_LOAD_OP_LOAD: Preserve the existing contents of the attachment

				• VK_ATTACHMENT_LOAD_OP_CLEAR: Clear the values to a constant at the start

				• VK_ATTACHMENT_LOAD_OP_DONT_CARE: Existing contents are undefined; we don’t care about them

				storeOp
				-------

				• VK_ATTACHMENT_STORE_OP_STORE: Rendered contents will be stored in memory and can be read later

				• VK_ATTACHMENT_STORE_OP_DONT_CARE: Contents of the framebuffer will be undefined after the rendering operation
			*/
			colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
			colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
			colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
			colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
			/*
				layouts
				-------

				• VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL: Images used as color attachment

				• VK_IMAGE_LAYOUT_PRESENT_SRC_KHR: Images to be presented in the swap chain

				• VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL: Images to be used as destination for a memory copy operation
			*/
			colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;//La imagen multimuestreada no se puede presentar directamente a VK_IMAGE_LAYOUT_PRESENT_SRC_KHR

			VkAttachmentReference colorAttachmentRef{};
			colorAttachmentRef.attachment = 0;	//El índice del array de attachment descriptions
			colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; //Que layout queremos que el attachment tenga durante un subpass
			
			/*Profundidad de la imagen*/
			VkAttachmentDescription depthAttachment{};
			depthAttachment.format = findDepthFormat();
			depthAttachment.samples = msaaSamples;
			depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
			depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
			depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
			depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
			depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

			VkAttachmentReference depthAttachmentRef{};
			depthAttachmentRef.attachment = 1;
			depthAttachmentRef.layout =	VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

			/*Resolución del color (para presentar la imagen multimuestreada)*/
			VkAttachmentDescription colorAttachmentResolve{};
			colorAttachmentResolve.format = swapChainImageFormat;
			colorAttachmentResolve.samples = VK_SAMPLE_COUNT_1_BIT;
			colorAttachmentResolve.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
			colorAttachmentResolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
			colorAttachmentResolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
			colorAttachmentResolve.stencilStoreOp =	VK_ATTACHMENT_STORE_OP_DONT_CARE;
			colorAttachmentResolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			colorAttachmentResolve.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

			VkAttachmentReference colorAttachmentResolveRef{};
			colorAttachmentResolveRef.attachment = 2;
			colorAttachmentResolveRef.layout =	VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

			/*
				El render pass puede componerse de varios subpass, cada paso subsecuente depende del estado del framebuffer en el paso previo. 
				Por ejemplo efectos de post-procesado que se aplican uno tras otro.

				Los subpass puede leer los siguiente tipos de attachment, en el ejemplo solo utilizamos el colorAttachment:

				• pInputAttachments: Attachments that are read from a shader
			
				• pResolveAttachments: Attachments used for multisampling color attachments

				• pDepthStencilAttachment: Attachment for depth and stencil data

				• pPreserveAttachments: Attachments that are not used by this subpass, but for which the data must be preserved
			*/
			VkSubpassDescription subpass{};
			subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
			subpass.colorAttachmentCount = 1;
			subpass.pColorAttachments = &colorAttachmentRef;
			subpass.pDepthStencilAttachment = &depthAttachmentRef;
			subpass.pResolveAttachments = &colorAttachmentResolveRef;

			/*
				VK_SUBPASS_EXTERNAL indica el subpass implicito antes o después del render pass.
				Dependiendo de en que parámetro se defina srcSubpass o dstSubpass.
			*/
			VkSubpassDependency dependency{};
			dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
			dependency.dstSubpass = 0;
			/*
				Esperamos en la etapa de color attachment output.
			*/
			dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			dependency.srcAccessMask = 0;
			/*
				Esperamos para transicionar hasta que iniciemos la etapa de color attachment write.
			*/
			dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

			std::array<VkAttachmentDescription, 3> attachments = { colorAttachment, depthAttachment, colorAttachmentResolve };

			//Definimos el render pass
			VkRenderPassCreateInfo renderPassInfo{};
			renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
			renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
			renderPassInfo.pAttachments = attachments.data();
			renderPassInfo.subpassCount = 1;
			renderPassInfo.pSubpasses = &subpass;
			renderPassInfo.dependencyCount = 1;
			renderPassInfo.pDependencies = &dependency;

			if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create render pass!");
			}
		}

	#pragma endregion

	#pragma region Framebuffers

		void createFramebuffers()
		{
			//Redimensionamos el vector al tamaño de swapChainImageViews
			swapChainFramebuffers.resize(swapChainImageViews.size());

			for (size_t i = 0; i < swapChainImageViews.size(); i++)
			{
				std::array<VkImageView, 3> attachments = { colorImageView, depthImageView, swapChainImageViews[i] };
				//Definimos el framebuffer
				VkFramebufferCreateInfo framebufferInfo{};
				framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
				framebufferInfo.renderPass = renderPass;
				framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
				framebufferInfo.pAttachments = attachments.data();
				framebufferInfo.width = swapChainExtent.width;
				framebufferInfo.height = swapChainExtent.height;
				framebufferInfo.layers = 1;

				if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS)
				{
					throw std::runtime_error("failed to create framebuffer!");
				}
			}
		}

	#pragma endregion

	#pragma region Comandos

		/*
			En vulkan los comandos no se ejecutan directamente con llamadas a funciones. En su lugar, tenemos que "grabar" los diferentes comandos en un command buffer.
			Esta forma de trabajar implica que la dificil tarea de configur los comandos se hace de antemano y puede definirse de tal forma que se ejecuten en múltiples hilos.

			Los comandos que grabemos, pueden reutilizarse tantas veces como queramos en el bucle principal y se destruyen explicitamente como el resto de los objetos al finalizar la ejecución.
		*/

		void createCommandPool()
		{
			QueueFamilyIndices queueFamilyIndices =	findQueueFamilies(physicalDevice);

			VkCommandPoolCreateInfo poolInfo{};
			poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
			poolInfo.queueFamilyIndex =	queueFamilyIndices.graphicsFamily.value();

			/*
				flags
				-------------

				• VK_COMMAND_POOL_CREATE_TRANSIENT_BIT: Hint that command buffers are rerecorded with new commands very often (may change memory allocation behavior)

				• VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT: Allow command buffers to be rerecorded individually, without this flag they all have to be reset together
			*/
			poolInfo.flags = 0; // Optional

			if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create command pool!");
			}
		}

		void createCommandBuffers()
		{
			//Redimensionamos el vector al tamaño de swapChainFramebuffers
			commandBuffers.resize(swapChainFramebuffers.size());

			//Alojamos el command buffer
			VkCommandBufferAllocateInfo allocInfo{};
			allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			allocInfo.commandPool = commandPool;
			/*
				Levels
				------

				• VK_COMMAND_BUFFER_LEVEL_PRIMARY: Can be submitted to a queue for execution, but cannot be called from other command buffers.

				• VK_COMMAND_BUFFER_LEVEL_SECONDARY: Cannot be submitted directly, but can be called from primary command buffers.
			*/
			allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

			if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to allocate command buffers!");
			}

			for (size_t i = 0; i < commandBuffers.size(); i++)
			{
				VkCommandBufferBeginInfo beginInfo{};
				beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
				/*
					flags
					-----
					• VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT: The command buffer will be rerecorded right after executing it once.

					• VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT: This is a secondary command buffer that will be entirely within a single render pass.

					• VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT: The command buffer can be resubmitted while it is also already pending execution.
				*/
				beginInfo.flags = 0; // Optional
				beginInfo.pInheritanceInfo = nullptr; // Optional

				if (vkBeginCommandBuffer(commandBuffers[i], &beginInfo) != VK_SUCCESS)
				{
					throw std::runtime_error("failed to begin recording command buffer!");
				}

				VkRenderPassBeginInfo renderPassInfo{};
				renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
				renderPassInfo.renderPass = renderPass;
				renderPassInfo.framebuffer = swapChainFramebuffers[i];

				//Extensión del area donde se va a renderizar
				renderPassInfo.renderArea.offset = { 0, 0 };
				renderPassInfo.renderArea.extent = swapChainExtent;

				//Los clearValues de VK_ATTACHMENT_LOAD_OP_CLEAR que utilizamos como operación de carga para ambos attachment
				std::array<VkClearValue, 2> clearValues = {};
				clearValues[0].color = { 0.0f, 0.0f, 0.0f, 1.0f };//Color negro, 100% opacidad
				clearValues[1].depthStencil = { 1.0f, 0 };
				renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
				renderPassInfo.pClearValues = clearValues.data();

				/*
					Subpass contents
					----------------
					Indica cómo se van a proporcionar los drawing commands durante el render pass:

					• VK_SUBPASS_CONTENTS_INLINE: The render pass commands will be embedded in the primary command buffer itself and no secondary command buffers will be executed.

					• VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS: The render pass commands will be executed from secondary command buffers.
				*/
				//Iniciamos el renderPass
				vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
				//Bindeamos el pipeline de gráficos
				vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

				//Bindeamos los vertex buffer indicando el array de buffers y los byte offsets desde los que empezar a leer los datos de los vértices
				VkBuffer vertexBuffers[] = { vertexBuffer };
				VkDeviceSize offsets[] = { 0 };
				vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, vertexBuffers, offsets);

				//Bindeamos el index buffer
				vkCmdBindIndexBuffer(commandBuffers[i], indexBuffer, 0,	VK_INDEX_TYPE_UINT32);
				//Bindeamos los descriptores, indicando el punto del pipeline, el layout, el array de descriptores y una serie de offsets
				vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[i], 0, nullptr);

				/*
					vkCmdDraw
					---------

					Recibe los siguiente parámetros:

					• commandBuffer

					• vertexCount: Even though we don’t have a vertex buffer, we technically still have 3 vertices to draw.

					• instanceCount: Used for instanced rendering, use 1 if you’re not doing that.

					• firstVertex: Used as an offset into the vertex buffer, defines the lowest value of gl_VertexIndex.

					• firstInstance: Used as an offset for instanced rendering, defines the lowest value of gl_InstanceIndex.
				*/
				//Dibujamos
				//vkCmdDraw(commandBuffers[i], static_cast<uint32_t>(vertices.size()), 1, 0, 0);
				vkCmdDrawIndexed(commandBuffers[i], static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

				//Finalizamos el renderPass
				vkCmdEndRenderPass(commandBuffers[i]);

				if (vkEndCommandBuffer(commandBuffers[i]) != VK_SUCCESS)
				{
					throw std::runtime_error("failed to record command buffer!");
				}
			}
		}

		VkCommandBuffer beginSingleTimeCommands()
		{
			//Alojamos un command buffer
			VkCommandBufferAllocateInfo allocInfo{};
			allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			allocInfo.commandPool = commandPool;
			allocInfo.commandBufferCount = 1;

			VkCommandBuffer commandBuffer;
			vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

			//Comenzamos a grabar el comando
			VkCommandBufferBeginInfo beginInfo{};
			beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;	//Indicamos al driver que vamos a utilizar el command buffer una única vez y esperar a que se termine la ejecución de la copia

			vkBeginCommandBuffer(commandBuffer, &beginInfo);

			return commandBuffer;
		}

		void endSingleTimeCommands(VkCommandBuffer commandBuffer)
		{
			vkEndCommandBuffer(commandBuffer);

			//Finalizamos la grabación
			VkSubmitInfo submitInfo{};
			submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers = &commandBuffer;

			vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
			vkQueueWaitIdle(graphicsQueue);

			vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
		}

	#pragma endregion

	#pragma region Buffers

		uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
		{
			VkPhysicalDeviceMemoryProperties memProperties;
			vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

			/*
				El parámetro typeFilter especifica el campo bit de los tipos de memoria que son adecuados.
				Podemos buscar el índice simplemente iterando por los tipos de memoria y comprobando que el bit correspondiente esté establecido en 1.

				Necesitamos que el tipo de memoria cumpla además las propiedades VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT y VK_MEMORY_PROPERTY_HOST_COHERENT_BIT para poder mapearla correctamente.
				Estas propiedades se indican en la estructura VkMemoryType.
			*/
			for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
			{
				if ((typeFilter & (1 << i) )
					&& (memProperties.memoryTypes[i].propertyFlags & properties) == properties) //Operación AND bit a bit 
				{
					return i;
				}
			}

			throw std::runtime_error("failed to find suitable memory type!");
		}

		void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory)
		{
			VkBufferCreateInfo bufferInfo{};
			bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
			bufferInfo.size = size;//Tamaño en bytes
			bufferInfo.usage = usage;
			bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; //El buffer puede ser compartido por varias colas (en este caso solo lo utiliza el graphicsQueue)
			bufferInfo.flags = 0;

			if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create buffer!");
			}

			/*
				La estructura se compone de los siguientes campos:

				• size: The size of the required amount of memory in bytes, may differ from bufferInfo.size.

				• alignment: The offset in bytes where the buffer begins in the allocated region of memory, depends on bufferInfo.usage and bufferInfo.flags.

				• memoryTypeBits: Bit field of the memory types that are suitable for the buffer.
			*/
			VkMemoryRequirements memRequirements;
			vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

			//Alojar la memoria
			VkMemoryAllocateInfo allocInfo{};
			allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
			allocInfo.allocationSize = memRequirements.size;
			allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

			if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to allocate buffer memory!");
			}

			vkBindBufferMemory(device, buffer, bufferMemory, 0);
		}

		void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size)
		{
			VkCommandBuffer commandBuffer = beginSingleTimeCommands();

			//Definimos el buffer copy
			VkBufferCopy copyRegion{};
			copyRegion.srcOffset = 0; // Optional
			copyRegion.dstOffset = 0; // Optional
			copyRegion.size = size;

			vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);
		
			endSingleTimeCommands(commandBuffer);
		}

		void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t	width, uint32_t height)
		{
			VkCommandBuffer commandBuffer = beginSingleTimeCommands();

			VkBufferImageCopy region{};
			region.bufferOffset = 0;
			region.bufferRowLength = 0;
			region.bufferImageHeight = 0;
		
			//Indican a qué parte de la imagen queremos copiar los pixels
			region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			region.imageSubresource.mipLevel = 0;
			region.imageSubresource.baseArrayLayer = 0;
			region.imageSubresource.layerCount = 1;
			region.imageOffset = { 0, 0, 0 };
			region.imageExtent = { width, height, 1 };

			vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

			endSingleTimeCommands(commandBuffer);
		}

	#pragma endregion

	#pragma region Vertex/Index data

		void createVertexBuffer()
		{
			VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

			//Utilizamos el staging buffer con la propiedad host visible para utilizarlo como buffer temporal
			VkBuffer stagingBuffer;
			VkDeviceMemory stagingBufferMemory;

			createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
		
			/*
				Indicamos el buffer origen y el destino:

				• VK_BUFFER_USAGE_TRANSFER_SRC_BIT: Buffer can be used as source in a memory transfer operation.

				• VK_BUFFER_USAGE_TRANSFER_DST_BIT: Buffer can be used as destination in a memory transfer operation.
			*/		

			/*
				Como puede que el driver no copie de los datos en la memoria del buffer de manera inmediat, Utilizamos un heap de memoria que sea coherente con el host.
				Esto es, usamos el parámetro VK_MEMORY_PROPERTY_HOST_COHERENT_BIT que hemos indicado antes al obtener los tipos de memoria adecuados.

				La transferencia de datos a la GPU ocurre en segundo plano.	La especificación garantiza que se completará en la siguiente llamada a vkQueueSubmit.
			*/
			void* data;
			vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);	//Mapeamos la memoria, accediendo a la región de la memoria deseada indicando offset y size
			memcpy(data, vertices.data(), (size_t) bufferSize);					//Copiamos los datos a la memoria del vertex buffer
			vkUnmapMemory(device, stagingBufferMemory);							//Deshacemos el mapeo

			//Utilizamos el vertex buffer para almacenar los datos que copiaremos desde el staging buffer
			createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT |	VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);

			//Copiamos el buffer origen en el destino
			copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

			//Destruimos el staging buffer de forma explicita y liberamos la memoria
			vkDestroyBuffer(device, stagingBuffer, nullptr);
			vkFreeMemory(device, stagingBufferMemory, nullptr);
		}

		void createIndexBuffer()
		{
			VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

			//Utilizamos el staging buffer con la propiedad host visible para utilizarlo como buffer temporal
			VkBuffer stagingBuffer;
			VkDeviceMemory stagingBufferMemory;

			createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

			/*
				Indicamos el buffer origen y el destino:

				• VK_BUFFER_USAGE_TRANSFER_SRC_BIT: Buffer can be used as source in a memory transfer operation.

				• VK_BUFFER_USAGE_TRANSFER_DST_BIT: Buffer can be used as destination in a memory transfer operation.
			*/

			/*
				Como puede que el driver no copie de los datos en la memoria del buffer de manera inmediat, Utilizamos un heap de memoria que sea coherente con el host.
				Esto es, usamos el parámetro VK_MEMORY_PROPERTY_HOST_COHERENT_BIT que hemos indicado antes al obtener los tipos de memoria adecuados.

				La transferencia de datos a la GPU ocurre en segundo plano.	La especificación garantiza que se completará en la siguiente llamada a vkQueueSubmit.
			*/
			void* data;
			vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);	//Mapeamos la memoria, accediendo a la región de la memoria deseada indicando offset y size
			memcpy(data, indices.data(), (size_t)bufferSize);					//Copiamos los datos a la memoria del index buffer
			vkUnmapMemory(device, stagingBufferMemory);							//Deshacemos el mapeo

			//Utilizamos el index buffer para almacenar los datos que copiaremos desde el staging buffer
			createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);

			//Copiamos el buffer origen en el destino
			copyBuffer(stagingBuffer, indexBuffer, bufferSize);

			//Destruimos el staging buffer de forma explicita y liberamos la memoria
			vkDestroyBuffer(device, stagingBuffer, nullptr);
			vkFreeMemory(device, stagingBufferMemory, nullptr);
		}

	#pragma endregion

	#pragma region Descriptores

		/*
			Vulkan utiliza descriptores de recursos, que permiten a los shaders acceder libremente a recursos definidos como buffers o imágenes.

			• Especificamos un descriptor layout durante la creación del pipeline de gráficos.
			• Alojamos un descriptor set desde un descriptor pool.
			• Bindeamos el descriptor set durante el renderizado.
		*/

		void createDescriptorSetLayout()
		{
			//Definimos el layout binding para el uniform buffer object (UBO)
			VkDescriptorSetLayoutBinding uboLayoutBinding{};
			uboLayoutBinding.binding = 0;
			uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; //UBO
			uboLayoutBinding.descriptorCount = 1;
			uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;//Etapa vertex shader
			uboLayoutBinding.pImmutableSamplers = nullptr; // Optional	//Es relevante para descriptores relacionados con el muestreo de imágenes

			//Definimos el layout binding para el sampler de texturas
			VkDescriptorSetLayoutBinding samplerLayoutBinding{};
			samplerLayoutBinding.binding = 1;
			samplerLayoutBinding.descriptorCount = 1;
			samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
			samplerLayoutBinding.pImmutableSamplers = nullptr;

			std::array<VkDescriptorSetLayoutBinding, 2> bindings = { uboLayoutBinding, samplerLayoutBinding };

			VkDescriptorSetLayoutCreateInfo layoutInfo{};
			layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
			layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
			layoutInfo.pBindings = bindings.data();

			if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create descriptor set layout!");
			}
		}

		void createUniformBuffers()
		{
			VkDeviceSize bufferSize = sizeof(UniformBufferObject);

			uniformBuffers.resize(swapChainImages.size());
			uniformBuffersMemory.resize(swapChainImages.size());

			for (size_t i = 0; i < swapChainImages.size(); i++)
			{
				createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
								uniformBuffers[i],	uniformBuffersMemory[i]);
			}
		}

		void updateUniformBuffer(uint32_t currentImage)
		{
			//Calculamos el tiempo transcurrido desde el inicio del renderizado en segundos, con precisión flotante.			
			static auto startTime = std::chrono::high_resolution_clock::now();

			auto currentTime = std::chrono::high_resolution_clock::now();
			float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();
				
			//Definimos a continuación el comportamiento de la transformación modelo, vista y proyección (MVP).
			/*
				Modelo
				------
				Se trata de una simple rotación sobre el eje Z utilizando la variable de tiempo (90º por segundo). 
			*/
			UniformBufferObject ubo{};
			ubo.model = glm::rotate(glm::scale(glm::mat4(1.0f), glm::vec3(0.075f, 0.075f, 0.075f)), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
			//ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
			/*
				Vista
				-----
				Definimos una vista superior con una inclinación de 45º.
			*/
			ubo.view = glm::lookAt(glm::vec3(6.0f, 6.0f, 6.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
			/*
				Proyección
				----------
				Definimos un campo de visión vertical de 45º.
			*/
			ubo.proj = glm::perspective(glm::radians(90.0f), swapChainExtent.width / (float)swapChainExtent.height, 0.1f, 10.0f);
			
			/* 
				La librería GLM fue diseñada para OpenGL, donde la coordenada Y de las coordenadas del clip esta invertida.
				La manera más facil para compensarla es cambiando el signo al eje Y en el factor de escala de la matriz de proyección. 
				De lo contrario, la imagen se renderizará volteada.
			*/
			ubo.proj[1][1] *= -1;

			void* data;
			vkMapMemory(device, uniformBuffersMemory[currentImage], 0, sizeof(ubo), 0, &data);//Mapeamos
			memcpy(data, &ubo, sizeof(ubo));//Copiamos los datos
			vkUnmapMemory(device, uniformBuffersMemory[currentImage]); //Deshacemos el mapeo
		}

		void createDescriptorPool()
		{
			std::array<VkDescriptorPoolSize, 2> poolSizes{};
			poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			poolSizes[0].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
			poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			poolSizes[1].descriptorCount =	static_cast<uint32_t>(swapChainImages.size());

			VkDescriptorPoolCreateInfo poolInfo{};
			poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
			poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
			poolInfo.pPoolSizes = poolSizes.data();
			poolInfo.maxSets = static_cast<uint32_t>(swapChainImages.size());//Número máximo de descriptores que pueden alojarse
			poolInfo.flags = 0;

			if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create descriptor pool!");
			}
		}

		void createDescriptorSets()
		{
			std::vector<VkDescriptorSetLayout> layouts(swapChainImages.size(), descriptorSetLayout);

			//Alojamos los descriptores
			VkDescriptorSetAllocateInfo allocInfo{};
			allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
			allocInfo.descriptorPool = descriptorPool;
			allocInfo.descriptorSetCount = static_cast<uint32_t> (swapChainImages.size()); //Un descriptor por cada imagen del swap chain
			allocInfo.pSetLayouts = layouts.data();

			descriptorSets.resize(swapChainImages.size());

			if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to allocate descriptor sets!");
			}

			for (size_t i = 0; i < swapChainImages.size(); i++)
			{
				VkDescriptorBufferInfo bufferInfo{};
				bufferInfo.buffer = uniformBuffers[i];
				bufferInfo.offset = 0;
				bufferInfo.range = sizeof(UniformBufferObject);

				VkDescriptorImageInfo imageInfo{};
				imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				imageInfo.imageView = textureImageView;
				imageInfo.sampler = textureSampler;

				//Definimos los sets de descriptores
				std::array<VkWriteDescriptorSet, 2> descriptorWrites{};

				descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				descriptorWrites[0].dstSet = descriptorSets[i];
				descriptorWrites[0].dstBinding = 0;
				descriptorWrites[0].dstArrayElement = 0;
				descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
				descriptorWrites[0].descriptorCount = 1;
				descriptorWrites[0].pBufferInfo = &bufferInfo;
				
				descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				descriptorWrites[1].dstSet = descriptorSets[i];
				descriptorWrites[1].dstBinding = 1;
				descriptorWrites[1].dstArrayElement = 0;
				descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				descriptorWrites[1].descriptorCount = 1;
				descriptorWrites[1].pImageInfo = &imageInfo;

				vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
			}
		}

	#pragma endregion

	#pragma region Image views

		/*
			La colección de objetos vkImageView que alimentan el swap chain.
			Se trata basicamente de un array que contiene la información de las imagenes.
		*/
		void createImageViews()
		{
			swapChainImageViews.resize(swapChainImages.size());

			for (uint32_t i = 0; i < swapChainImages.size(); i++)
			{
				swapChainImageViews[i] = createImageView(swapChainImages[i], swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
			}
		}

		VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, uint32_t mipLevels)
		{
			VkImageViewCreateInfo viewInfo{};
			viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			viewInfo.image = image;
			viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
			viewInfo.format = format;
			//Apuntamos solamente al color de la imagen
			viewInfo.subresourceRange.aspectMask = aspectFlags;
			//Nivel de mipmap
			viewInfo.subresourceRange.baseMipLevel = 0;
			viewInfo.subresourceRange.levelCount = mipLevels; 
			//Capas múltiples (3D estereoscópico)
			viewInfo.subresourceRange.baseArrayLayer = 0;
			viewInfo.subresourceRange.layerCount = 1;

			VkImageView imageView;
			if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create image view!");
			}

			return imageView;
		}

	#pragma endregion

	#pragma region Imágenes

		void createTextureImage()
		{
			int texWidth, texHeight, texChannels;
			//STBI_rgb_alpha fuerza a que la imagen se cargue añadiendo un canal alfa
			stbi_uc* pixels = stbi_load(TEXTURE_PATH.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);

			VkDeviceSize imageSize = texWidth * texHeight * 4; //4 bytes por pixel en el caso de STBI_rgb_alpha

			/*
				Calculamos miplevels
				--------------------

				Utilizamos max para obtener la dimensión máxima y log2 para saber cuantas veces es divisible entre 2.
				La función floor para cuando el número no sea una potencia de 2. Añadimos el 1 del final para que la imagen tenga como mínimo un nivel de mipmap.
			*/
			mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1;

			VkBuffer stagingBuffer;
			VkDeviceMemory stagingBufferMemory;

			//Creamos el buffer como host visible y como fuente para copiarlo a posteriori
			createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |	VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
						stagingBuffer, stagingBufferMemory);

			//Mapeamos la memoria, copiamos la imagen y deshacemos el mapeo
			void* data;
			vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
			memcpy(data, pixels, static_cast<size_t>(imageSize));
			vkUnmapMemory(device, stagingBufferMemory);

			//Liberamos el array de píxeles
			stbi_image_free(pixels);

			//Creamos la imagen
			createImage(texWidth, texHeight, mipLevels, VK_SAMPLE_COUNT_1_BIT, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
						VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
						textureImage, textureImageMemory);
			
			//Transicionamos el layout de la imagen
			transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1);

			//Copiamos el buffer a la imagen
			copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));

			generateMipmaps(textureImage, VK_FORMAT_R8G8B8A8_SRGB, texWidth, texHeight, mipLevels);

			//Necesitamos una transición más para empezar a muestrear la imagen en el shader
			//Vamos a transicionar a VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL cuando generamos los mipmaps
			//transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, mipLevels);

			//Destruimos el staging buffer de forma explicita y liberamos la memoria
			vkDestroyBuffer(device, stagingBuffer, nullptr);
			vkFreeMemory(device, stagingBufferMemory, nullptr);
		}

		void createImage(uint32_t width, uint32_t height, uint32_t mipLevels,
						VkSampleCountFlagBits numSamples, VkFormat format, VkImageTiling tiling,
						VkImageUsageFlags usage, VkMemoryPropertyFlags properties,
						VkImage& image, VkDeviceMemory& imageMemory)
		{
			//Definimos la imagen
			VkImageCreateInfo imageInfo{};
			imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
			imageInfo.imageType = VK_IMAGE_TYPE_2D;
			imageInfo.extent.width = static_cast<uint32_t>(width);
			imageInfo.extent.height = static_cast<uint32_t>(height);
			imageInfo.extent.depth = 1; //Se trata de una imagen de dos dimensiones
			imageInfo.mipLevels = mipLevels;
			imageInfo.arrayLayers = 1;
			imageInfo.format = format; //Utilizamos el mismo formato que tienen los pixels del buffer para los texels de la imagen
			/*
				Tiling
				------

				• VK_IMAGE_TILING_LINEAR: Texels are laid out in row-major order like our pixels array

				• VK_IMAGE_TILING_OPTIMAL: Texels are laid out in an implementation defined order for optimal access
			*/
			imageInfo.tiling = tiling;
			/*
				Layout inicial
				--------------
				• VK_IMAGE_LAYOUT_UNDEFINED: Not usable by the GPU and the very first transition will discard the texels.

				• VK_IMAGE_LAYOUT_PREINITIALIZED: Not usable by the GPU, but the first transition will preserve the texels.
			*/
			imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			//La imagen se va a utilizar como el destino para la copia del buffer, además queremos que sea accesible desde el shader para colorearla
			imageInfo.usage = usage;
			imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; //Solo utilizable por una única cola
			imageInfo.samples = numSamples;
			imageInfo.flags = 0; // Optional

			if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create image!");
			}

			//Alojamos la memoria para la imagen
			VkMemoryRequirements memRequirements;
			vkGetImageMemoryRequirements(device, image, &memRequirements);

			VkMemoryAllocateInfo allocInfo{};
			allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
			allocInfo.allocationSize = memRequirements.size;
			allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

			if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to allocate image memory!");
			}

			vkBindImageMemory(device, image, imageMemory, 0);
		}

		void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels)
		{
			VkCommandBuffer commandBuffer = beginSingleTimeCommands();

			VkImageMemoryBarrier barrier{};
			barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			barrier.oldLayout = oldLayout;
			barrier.newLayout = newLayout;
			//Para transferir la propiedad de una familia de cola
			barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.image = image;
			barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			barrier.subresourceRange.baseMipLevel = 0;
			barrier.subresourceRange.levelCount = mipLevels;
			barrier.subresourceRange.baseArrayLayer = 0;
			barrier.subresourceRange.layerCount = 1;

			//Indican qué operaciones deben ocurrir antes de la barrera y qué operaciones deben esperar en la barrera
			VkPipelineStageFlags sourceStage;
			VkPipelineStageFlags destinationStage;

			/*
				Existe un layout general, llamado VK_IMAGE_LAYOUT_GENERAL, para cuando se utiliza la imagen como input y output por ejemplo.
				Sin embargo este layout no ofrece el mejor rendimiento. Por tanto es más efectivo llevar un control explícito de los layouts según el tratamiento que le vamos a dar a la imagen.

			*/
			if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
			{
				barrier.srcAccessMask = 0;
				barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

				sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
				destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT; //Realmente se trata de una pseudo etapa dentro del pipeline
			}
			else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
			{
				barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
				barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
			
				sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
				destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
			}
			else
			{
				throw std::invalid_argument("unsupported layout transition!");
			}
				
			//Los parámetros srcStageMask y dstStageMask dependen de cómo se va a utilizar el recurso antes de la barrera y después de la barrera
			vkCmdPipelineBarrier(commandBuffer, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);

			endSingleTimeCommands(commandBuffer);
		}

		void createTextureImageView()
		{
			textureImageView = createImageView(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT, mipLevels);
		}

		void generateMipmaps(VkImage image, VkFormat imageFormat, uint32_t texWidth, uint32_t texHeight, uint32_t mipLevels)
		{
			//Comprobamos si el formato de imagen soporta blitting lineal
			VkFormatProperties formatProperties;
			vkGetPhysicalDeviceFormatProperties(physicalDevice, imageFormat, &formatProperties);

			if (!(formatProperties.optimalTilingFeatures &	VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT))
			{
				throw std::runtime_error("texture image format does not support	linear blitting!");
			}

			VkCommandBuffer commandBuffer = beginSingleTimeCommands();

			VkImageMemoryBarrier barrier{};
			barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			barrier.image = image;
			barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			barrier.subresourceRange.baseArrayLayer = 0;
			barrier.subresourceRange.layerCount = 1;
			barrier.subresourceRange.levelCount = 1;
			
			uint32_t mipWidth = texWidth;
			uint32_t mipHeight = texHeight;

			for (uint32_t i = 1; i < mipLevels; i++)
			{
				barrier.subresourceRange.baseMipLevel = i - 1;
				barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
				barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
				barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
				barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
				
				vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,	0, 0, nullptr, 0, nullptr, 1, &barrier);

				VkImageBlit blit{};
				blit.srcOffsets[0] = { 0, 0, 0 };
				//la dimensión del eje Z es 1 ya que se trata de una imagen 2D
				blit.srcOffsets[1] = { static_cast<int32_t>(mipWidth), static_cast<int32_t>(mipHeight), 1 }; 
				blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
				blit.srcSubresource.mipLevel = i - 1; //Nivel mip de origen
				blit.srcSubresource.baseArrayLayer = 0;
				blit.srcSubresource.layerCount = 1;
				blit.dstOffsets[0] = { 0, 0, 0 };
				//Dividimos entre 2, la dimensión del eje Z es 1 ya que se trata de una imagen 2D
				blit.dstOffsets[1] = { static_cast<int32_t>(mipWidth > 1 ? mipWidth / 2 : 1), static_cast<int32_t>(mipHeight >	1 ? mipHeight / 2 : 1), 1 };
				blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
				blit.dstSubresource.mipLevel = i; //Nivel mip de destino
				blit.dstSubresource.baseArrayLayer = 0;
				blit.dstSubresource.layerCount = 1;

				vkCmdBlitImage(commandBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);

				barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
				barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
				barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

				vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

				if (mipWidth > 1) mipWidth /= 2;
				if (mipHeight > 1) mipHeight /= 2;
			}

			//El último nivel de mip
			barrier.subresourceRange.baseMipLevel = mipLevels - 1;
			barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
			
			vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,	VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
			
			endSingleTimeCommands(commandBuffer);
		}

	#pragma endregion

	#pragma region Samplers

		void createTextureSampler()
		{
			//Definimos el sampler
			VkSamplerCreateInfo samplerInfo{};
			samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
			//Describen cómo interpolar los texels que estan magnificados(oversampling) o minificados(undersampling)
			samplerInfo.magFilter = VK_FILTER_LINEAR;
			samplerInfo.minFilter = VK_FILTER_LINEAR;
			/*
				Modos de direccionamiento
				------------

				• VK_SAMPLER_ADDRESS_MODE_REPEAT: Repeat the texture when going beyond the image dimensions.

				• VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT: Like repeat, but inverts the coordinates to mirror the image when going beyond the dimensions.

				• VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE: Take the color of the edge closest to the coordinate beyond the image dimensions.

				• VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE: Like clamp to edge, but instead uses the edge opposite to the closest edge.

				• VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER: Return a solid color when sampling beyond the dimensions of the image.
			*/
			samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
			samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
			samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
			samplerInfo.anisotropyEnable = VK_TRUE;
			samplerInfo.maxAnisotropy = 16;
			samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
			samplerInfo.unnormalizedCoordinates = VK_FALSE; //Coordenadas normalizadas.Para poder utilizar texturas de diferente resolución con las mismas coordenadas
			//Para comparar los texels a un valor. El resultado de esta comparación se utiliza en operaciones de filtrado, por ejemplo en shadow maps.
			samplerInfo.compareEnable = VK_FALSE;
			samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
			samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
			samplerInfo.mipLodBias = 0.0f;
			samplerInfo.minLod = 0.0f;
			samplerInfo.maxLod = static_cast<float>(mipLevels);

			if (vkCreateSampler(device, &samplerInfo, nullptr, &textureSampler) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create texture sampler!");
			}
		}

	#pragma endregion

	#pragma region Depth images

		/*
			Para controlar la profundidad de las imágenes podemos utilizar un depth buffer generado con el depth attachment de las imagenes. Que contiene la información acerca de la profundidad.
		*/

		VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features)
		{
			for (VkFormat format : candidates)
			{
				/*
					La estructura contiene:

					• linearTilingFeatures: Use cases that are supported with linear tiling
					
					• optimalTilingFeatures: Use cases that are supported with optimal tiling

					• bufferFeatures: Use cases that are supported for buffers
				*/
				VkFormatProperties props;
				vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);

				if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures& features) == features)
				{
					return format;
				}
				else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features)
				{
					return format;
				}
			}		

			throw std::runtime_error("failed to find supported format!");
		}

		VkFormat findDepthFormat()
		{
			return findSupportedFormat({ VK_FORMAT_D32_SFLOAT,
										VK_FORMAT_D32_SFLOAT_S8_UINT,
										VK_FORMAT_D24_UNORM_S8_UINT },
										VK_IMAGE_TILING_OPTIMAL,
										VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
		}

		bool hasStencilComponent(VkFormat format)
		{
			return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
		}

		void createDepthResources()
		{
			VkFormat depthFormat = findDepthFormat();

			createImage(swapChainExtent.width, swapChainExtent.height, 1,
						msaaSamples, depthFormat, VK_IMAGE_TILING_OPTIMAL,
						VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
						VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
						depthImage, depthImageMemory);

			depthImageView = createImageView(depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT, 1);
		}


	#pragma endregion

	#pragma region Multimuestreo
		
		/*
			Tenemos que obtener el número de muestras para el color y para la profundidad (depth) ya que estamos usando un depth buffer
		*/
		VkSampleCountFlagBits getMaxUsableSampleCount()
		{
			VkPhysicalDeviceProperties physicalDeviceProperties;
			vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);

			VkSampleCountFlags counts = physicalDeviceProperties.limits.framebufferColorSampleCounts & physicalDeviceProperties.limits.framebufferDepthSampleCounts;
			
			if (counts & VK_SAMPLE_COUNT_64_BIT)
			{
				return	VK_SAMPLE_COUNT_64_BIT;
			}
			if (counts & VK_SAMPLE_COUNT_32_BIT)
			{
				return VK_SAMPLE_COUNT_32_BIT;
			}
			if (counts & VK_SAMPLE_COUNT_16_BIT)
			{
				return VK_SAMPLE_COUNT_16_BIT;
			}
			if (counts & VK_SAMPLE_COUNT_8_BIT)
			{
				return	VK_SAMPLE_COUNT_8_BIT;
			}
			if (counts & VK_SAMPLE_COUNT_4_BIT)
			{
				return VK_SAMPLE_COUNT_4_BIT;
			}
			if (counts & VK_SAMPLE_COUNT_2_BIT)
			{
				return VK_SAMPLE_COUNT_2_BIT;
			}
			
			return VK_SAMPLE_COUNT_1_BIT;
		}

		void createColorResources()
		{
			VkFormat colorFormat = swapChainImageFormat;

			createImage(swapChainExtent.width, swapChainExtent.height, 1,
						msaaSamples, colorFormat, VK_IMAGE_TILING_OPTIMAL,
						VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT |
						VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
						VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
						colorImage,	colorImageMemory);

			colorImageView = createImageView(colorImage, colorFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
		}

	#pragma endregion

	#pragma region Modelos
		
		void loadModel()
		{
			//Cargamos el modelo
			tinyobj::attrib_t attrib;
			std::vector<tinyobj::shape_t> shapes;
			std::vector<tinyobj::material_t> materials;
			std::string warn, err;
			
			if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, MODEL_PATH.c_str()))
			{
				throw std::runtime_error(warn + err);
			}

			/*
				LoadObj incluye un parámetro opcional para triangular automaticamente las caras (faces) del modelo.
				Como nuestra aplicación solo renderiza triangulos, esta conversión nos facilita la carga del modelo.
			*/
			for (const auto& shape : shapes)
			{
				for (const auto& index : shape.mesh.indices)
				{
					std::unordered_map<Vertex, uint32_t> uniqueVertices{};
					Vertex vertex{};

					
					/*
						attrib.vertices es un array de valores float, por eso es necesario multiplicar el índice por 3
						El offset del final (0, 1, 2) sirve para acceder a los componentes X, Y, Z o en el caso de coordenadas de texturas a los componentes U y V
					*/
					vertex.pos = { attrib.vertices[3 * index.vertex_index + 0],
									attrib.vertices[3 * index.vertex_index + 1],
									attrib.vertices[3 * index.vertex_index + 2] };
										
					vertex.texCoord = { attrib.texcoords[2 * index.texcoord_index + 0],
										1.0f - attrib.texcoords[2 * index.texcoord_index + 1] };

					vertex.color = { 1.0f, 1.0f, 1.0f };

					if (uniqueVertices.count(vertex) == 0)
					{
						uniqueVertices[vertex] = static_cast<uint32_t>(vertices.size());
						vertices.push_back(vertex);
					}

					indices.push_back(static_cast<uint32_t>(indices.size()));
				}
			}
		}

	#pragma endregion

	#pragma endregion

};

int main()
{
	HelloTriangleApplication app;

	try
	{
		app.run();
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}