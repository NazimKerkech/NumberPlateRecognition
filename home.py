from multiapp import MultiApp
import conception, detections, dataset

app = MultiApp()

app.add_app("Conception", conception.app)
app.add_app("Dataset", dataset.app)
app.add_app("Detection", detections.app)

app.run()
