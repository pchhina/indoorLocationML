library(shiny)
library(ggplot2)

ui <- fluidPage(
                sidebarLayout(
                              sidebarPanel(
                                           actionButton("rand", "Locate")
                              ),
                              mainPanel(plotOutput(outputId = "point"))
                ))

server <- function(input, output) {
    sampleData <- eventReactive(input$rand, 
                                testingData[sample(1:1111,1), 1:522])
    output$point <- renderPlot({
    obsLong <- sampleData()$LONGITUDE
    predLong <- predict(rfFit,sampleData())
    df <- data.frame(cat = c("obs", "pred"),
                     long = c(obsLong, predLong),
                     lat = c(sampleData()$LATITUDE, sampleData()$LATITUDE))
    ggplot(data = df, aes(long, lat)) +
               geom_point(shape = 21,
                          color = "yellow",
                          fill = "black",
                          size = 5,
                          stroke = 2) +
               scale_x_continuous(limits = c(-7695.9, -7299.9)) +
               scale_y_continuous(limits = c(4864745.7, 4865017.4)) +
               theme(panel.background = element_rect(fill = "black"),
                     panel.grid.major = element_blank(),
                     panel.grid.minor = element_line(color = "grey"))
    })
}

shinyApp(ui = ui, server = server)
